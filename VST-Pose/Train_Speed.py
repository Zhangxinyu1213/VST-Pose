############   alpha is important   ###########
alpha = 0.2
######################
import os
import argparse
import yaml
import torch.optim as optim
from torch import nn
from Model.Speed.conv_STFormer_Speed import CSIToKeypointModel
from Feeder.mmfi_lib.mmfi import make_dataset, make_dataloader
from tqdm import tqdm
import time
from utils import *
from torch.utils.tensorboard import SummaryWriter


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('conv_STFormer_Speed')
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default=r'config/config.yaml')
    args = parser.parse_args()

    ####################### Load Config ####################
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    dataset_name = config['dataset_name']
    dataset_root = config['dataset_root']
    if dataset_name == 'mmfi-csi':
        print(dataset_name, config['protocol'])

    train_batchsize = config['train_loader']['batch_size']
    val_batchsize = config['validation_loader']['batch_size']

    ################### Dataset Setup ###################
    if dataset_name == 'mmfi-csi':
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['split_to_use'])
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        # Create dataloaders
        train_dataset, val_dataset = make_dataset(dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator,
                                       **config['train_loader'])
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator,
                                     **config['validation_loader'])
    else:
        print('dataset_name_error')

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ################### Model Creation ###################
    if dataset_name == 'mmfi-csi':
        model = CSIToKeypointModel(num_frames=config['window_size']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['base_learning_rate'], betas=(0.9, 0.999))
        criterion = nn.MSELoss()  # Mean Squared Error loss for keypoint prediction
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    print('Learning_rate:{}'.format(config['base_learning_rate']))

    ################### Start Training ###################
    num_epochs = config['total_epoch']
    writer = SummaryWriter("logs/Pose_Training")
    best_val_loss = float('inf')
    best_val_pck = [0.0] * 5  # For different PCK thresholds

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        ################### Training ###################
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", ncols=100)
        for batch_idx, batch_data in enumerate(pbar):
            input_wifi_csi = batch_data['input_wifi-csi'].to(device)
            input_rgb = batch_data['output'].to(device)  # 17*3
            kp = input_rgb
            kp_speed = kp[:, -1, :, :] - kp[:, 0, :, :]  # (b, 17, 3)
            optimizer.zero_grad()
            output, speed = model(input_wifi_csi)
            loss = (1 - alpha) * criterion(output, kp) + alpha * criterion(speed, kp_speed)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % 30 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        scheduler.step()
        print(f"Epoch {epoch + 1} - Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        ################### Validation ###################
        model.eval()
        val_loss = 0.0
        pck_iter = [[] for _ in range(5)]
        mpjpe_list = []
        pampjpe_list = []

        pbar_test = tqdm(val_loader, desc=f"Testing Epoch {epoch + 1}", ncols=100)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar_test):
                input_wifi_csi = batch_data['input_wifi-csi'].to(device)
                input_rgb = batch_data['output'].to(device)
                kp = input_rgb
                kp_speed = kp[:, -1, :, :] - kp[:, 0, :, :]  # (b, 17, 3)
                output, speed = model(input_wifi_csi)
                loss = (1 - alpha) * criterion(output, kp) + alpha * criterion(speed, kp_speed)
                val_loss += loss.item()

                predicted_val_pose = output.view(-1, 17, 3)
                val_pose_gt = input_rgb.view(-1, 17, 3)
                mpjpe, pampjpe = calulate_error(predicted_val_pose.data.cpu().numpy(), val_pose_gt.data.cpu().numpy(),
                                                align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()

                # Compute PCK
                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(predicted_val_pose.permute(0, 2, 1).data.cpu().numpy(),
                                                          val_pose_gt.permute(0, 2, 1).data.cpu().numpy(), percentage,
                                                          align=False, dataset=dataset_name))

        val_loss /= len(val_loader)
        avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)

        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)

        # Calculate and log PCK
        if dataset_name == 'mmfi-csi':
            pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]
        elif dataset_name == 'wipose':
            pck_overall = [np.mean(pck_value, 0)[18] for pck_value in pck_iter]

        for idx, percentage in enumerate([50, 40, 30, 20, 10]):
            writer.add_scalar(f"PCK/test_pck{percentage}", pck_overall[idx], epoch + 1)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1} took {epoch_time:.2f} seconds")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f'avg_val_mpjpe:{avg_val_mpjpe}  ', f'avg_val_pampjpe:  {avg_val_pampjpe}')
        print(
            f' test pck50: {pck_overall[0]}, test pck40: {pck_overall[1]}, test pck30: {pck_overall[2]}, test pck20: {pck_overall[3]}, test pck10: {pck_overall[4]}.')

        ################### Save Best Model ###################
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'Saving best model with val_loss {best_val_loss:.4f} at epoch {epoch + 1}')
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            print(f'saving best model with mpjpe {best_val_mpjpe} at {epoch} epoch!')
        if avg_val_pampjpe < best_val_pampjpe:
            best_val_pampjpe = avg_val_pampjpe
            print(f'saving best model with pampjpe {best_val_pampjpe} at {epoch} epoch!')

        for idx, pck_value in enumerate(pck_overall):
            if pck_value > best_val_pck[idx]:
                best_val_pck[idx] = pck_value
                print(
                    f'Saving best model with pck{[50, 40, 30, 20, 10][idx]} {best_val_pck[idx]:.4f} at epoch {epoch + 1}')

        # Save result summary
        summary_file_path = os.path.join(weights_path, "result_summary.txt")
        with open(summary_file_path, "w") as file:
            file.write("Best Results Summary:\n")
            file.write(f"Best MPJPE: {best_val_mpjpe:.4f}\n")
            file.write(f"Best PA-MPJPE: {best_val_pampjpe:.4f}\n")
            for idx, pck_value in enumerate(best_val_pck):
                file.write(f"Best PCK@{[50, 40, 30, 20, 10][idx]}: {pck_value:.4f}\n")

        print(f"Best results have been saved to {summary_file_path}")
