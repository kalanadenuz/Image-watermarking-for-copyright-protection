"""
train.py
Training script for the image watermarking encoder-decoder system.
Trains both encoder and decoder simultaneously with combined loss function.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import custom modules
from models.encoder import get_encoder
from models.decoder import get_decoder
from dataset import create_dataloaders
from attacks import apply_random_attack, apply_combined_attacks


class WatermarkingSystem:
    """
    Complete watermarking system with encoder and decoder.
    Handles training, validation, and checkpoint saving.
    """
    def __init__(self, config):
        """
        Initialize the watermarking system.
        
        Args:
            config (dict): Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        # Create models
        print("\nInitializing models...")
        self.encoder = get_encoder(
            watermark_size=config['watermark_size'],
            deep=config.get('deep_encoder', False)
        ).to(self.device)
        
        self.decoder = get_decoder(
            watermark_size=config['watermark_size'],
            architecture=config.get('decoder_arch', 'standard')
        ).to(self.device)
        
        # Count parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"Encoder parameters: {encoder_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        print(f"Total parameters: {encoder_params + decoder_params:,}")
        
        # Create optimizers
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.encoder_optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.decoder_optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()  # For image reconstruction
        self.bce_loss = nn.BCELoss()  # For watermark extraction
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def compute_loss(self, original_image, watermarked_image, original_watermark, extracted_watermark):
        """
        Compute combined loss for encoder and decoder.
        
        Args:
            original_image: Original cover image
            watermarked_image: Image with embedded watermark
            original_watermark: Original watermark
            extracted_watermark: Extracted watermark from decoder
        
        Returns:
            tuple: (total_loss, image_loss, watermark_loss)
        """
        # Image reconstruction loss (watermarked should look like original)
        image_loss = self.mse_loss(watermarked_image, original_image)
        
        # Watermark extraction loss (extracted should match original)
        watermark_loss = self.bce_loss(extracted_watermark, original_watermark)
        
        # Combined loss with weights
        image_weight = self.config['image_loss_weight']
        watermark_weight = self.config['watermark_loss_weight']
        
        total_loss = image_weight * image_loss + watermark_weight * watermark_loss
        
        return total_loss, image_loss, watermark_loss
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            dict: Dictionary of average losses
        """
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        total_image_loss = 0
        total_watermark_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        
        for batch_idx, (images, watermarks) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            watermarks = watermarks.to(self.device)
            
            # Forward pass through encoder
            watermarked_images = self.encoder(images, watermarks)
            
            # Apply random attacks to watermarked images
            if self.config['use_attacks']:
                attacked_images = apply_random_attack(
                    watermarked_images,
                    attack_prob=self.config['attack_probability']
                )
            else:
                attacked_images = watermarked_images
            
            # Forward pass through decoder
            extracted_watermarks = self.decoder(attacked_images)
            
            # Compute loss
            loss, image_loss, watermark_loss = self.compute_loss(
                images, watermarked_images, watermarks, extracted_watermarks
            )
            
            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
            
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_image_loss += image_loss.item()
            total_watermark_loss += watermark_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'img': f'{image_loss.item():.4f}',
                'wm': f'{watermark_loss.item():.4f}'
            })
            
            # Log to tensorboard (every N batches)
            if batch_idx % 10 == 0:
                step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/ImageLoss', image_loss.item(), step)
                self.writer.add_scalar('Train/WatermarkLoss', watermark_loss.item(), step)
        
        # Calculate average losses
        num_batches = len(train_loader)
        avg_losses = {
            'loss': total_loss / num_batches,
            'image_loss': total_image_loss / num_batches,
            'watermark_loss': total_watermark_loss / num_batches
        }
        
        return avg_losses
    
    def validate(self, val_loader, epoch):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            dict: Dictionary of average validation losses
        """
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        total_image_loss = 0
        total_watermark_loss = 0
        
        with torch.no_grad():
            for images, watermarks in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                watermarks = watermarks.to(self.device)
                
                # Forward pass
                watermarked_images = self.encoder(images, watermarks)
                
                # Apply attacks for robustness testing
                if self.config['use_attacks']:
                    attacked_images = apply_random_attack(watermarked_images, attack_prob=0.8)
                else:
                    attacked_images = watermarked_images
                
                extracted_watermarks = self.decoder(attacked_images)
                
                # Compute loss
                loss, image_loss, watermark_loss = self.compute_loss(
                    images, watermarked_images, watermarks, extracted_watermarks
                )
                
                total_loss += loss.item()
                total_image_loss += image_loss.item()
                total_watermark_loss += watermark_loss.item()
        
        # Calculate average losses
        num_batches = len(val_loader)
        avg_losses = {
            'loss': total_loss / num_batches,
            'image_loss': total_image_loss / num_batches,
            'watermark_loss': total_watermark_loss / num_batches
        }
        
        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_losses['loss'], epoch)
        self.writer.add_scalar('Val/ImageLoss', avg_losses['image_loss'], epoch)
        self.writer.add_scalar('Val/WatermarkLoss', avg_losses['watermark_loss'], epoch)
        
        return avg_losses
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        
        for epoch in range(self.current_epoch + 1, self.config['num_epochs'] + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config['num_epochs']}")
            print(f"{'='*60}")
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"\nTraining - Loss: {train_losses['loss']:.4f}, "
                  f"Image Loss: {train_losses['image_loss']:.4f}, "
                  f"Watermark Loss: {train_losses['watermark_loss']:.4f}")
            
            # Validate
            val_losses = self.validate(val_loader, epoch)
            print(f"Validation - Loss: {val_losses['loss']:.4f}, "
                  f"Image Loss: {val_losses['image_loss']:.4f}, "
                  f"Watermark Loss: {val_losses['watermark_loss']:.4f}")
            
            # Update learning rate schedulers
            self.encoder_scheduler.step(val_losses['loss'])
            self.decoder_scheduler.step(val_losses['loss'])
            
            # Save checkpoint
            is_best = val_losses['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['loss']
            
            if epoch % self.config['save_frequency'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)
        
        self.writer.close()


def main():
    """
    Main function to set up and run training.
    """
    parser = argparse.ArgumentParser(description='Train image watermarking system')
    parser.add_argument('--image_dir', type=str, default='data/images', help='Directory with cover images')
    parser.add_argument('--watermark_dir', type=str, default='data/watermarks', help='Directory with watermarks')
    parser.add_argument('--checkpoint_dir', type=str, default='models', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--watermark_size', type=int, default=32, help='Watermark size')
    parser.add_argument('--image_loss_weight', type=float, default=1.0, help='Weight for image loss')
    parser.add_argument('--watermark_loss_weight', type=float, default=1.0, help='Weight for watermark loss')
    parser.add_argument('--use_attacks', action='store_true', help='Use attacks during training')
    parser.add_argument('--attack_probability', type=float, default=0.5, help='Probability of attack')
    parser.add_argument('--deep_encoder', action='store_true', help='Use deep encoder architecture')
    parser.add_argument('--decoder_arch', type=str, default='standard', choices=['standard', 'deep', 'attention'], help='Decoder architecture')
    parser.add_argument('--save_frequency', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create configuration dictionary
    config = vars(args)
    
    # Save configuration
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        image_dir=args.image_dir,
        watermark_dir=args.watermark_dir,
        image_size=args.image_size,
        watermark_size=args.watermark_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create watermarking system
    system = WatermarkingSystem(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        system.load_checkpoint(args.resume)
    
    # Start training
    system.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
