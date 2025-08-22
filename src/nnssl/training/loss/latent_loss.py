from loguru import logger
import numpy as np 
from nnssl.training.loss.mse_loss import LossMaskMSELoss, MAEMSELoss
import torch
from torch import nn
import torch.nn.functional as F





class NaiveProjector(nn.Module):
    """Original direct projection approach"""
    def __init__(self, bottleneck_dim, subject_dim):
        super().__init__()
        bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))    
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, subject_dim * 2),  # Intermediate layer
            nn.LayerNorm(subject_dim * 2),
            nn.ReLU(),
            nn.Linear(subject_dim * 2, subject_dim),     # Target: subject_dim
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Standard flattening
        bottleneck_flat = latent.view(batch_size, -1)  # [batch, 40000]
        return self.projector(bottleneck_flat)


class ProgressiveProjector(nn.Module):
    """Progressive reduction approach"""
    def __init__(self, bottleneck_dim, subject_dim):
        super().__init__()
        bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, 2048),     
            nn.LayerNorm(2048),
            nn.ReLU(),
            #nn.Dropout(0.1), Regularitation probably not needed here
            
            nn.Linear(2048, 512),                 
            nn.LayerNorm(512),
            nn.ReLU(),
            #nn.Dropout(0.1),
            
            nn.Linear(512, subject_dim),          
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Standard flattening
        bottleneck_flat = latent.view(batch_size, -1)  
        return self.projector(bottleneck_flat)



class PooledProgressiveProjector(nn.Module):
    """Spatial pooling + progressive reduction with channel portion selection"""
    
    def __init__(self, bottleneck_dim, subject_dim, channel_portion=1.0):
        """
        Args:
            bottleneck_dim: Original bottleneck dimensions as shape (e.g., [320, 5, 5, 5])
            subject_dim: Target embedding dimension
            channel_portion: Portion of channels to use (0.0 to 1.0)
                - 0.0: Use 1 channel (minimum to avoid errors)
                - 1.0: Use all channels
                - 0.5: Use half the channels (first half)
        """
        super().__init__()
        
        # For compatibility with bottleneck_flat_dim = np.prod(np.array(bottleneck_dim))
        if isinstance(bottleneck_dim, (list, tuple)):
            self.bottleneck_shape = bottleneck_dim
            total_channels = bottleneck_dim[0]  # 320
            self.spatial_dims = bottleneck_dim[1:]  # [5, 5, 5]
        else:
            # If it's already flattened dimension, assume default shape
            raise ValueError("bottleneck_dim should be the shape [channels, d, h, w], not flattened dimension")
        
        # Calculate number of channels to use based on portion
        self.channel_portion = max(0.0, min(1.0, channel_portion))  # Clamp between 0 and 1
        n_channels_to_use = max(1, int(total_channels * self.channel_portion))  # At least 1 channel
        
        # Select first n_channels_to_use channels
        self.selected_channels = list(range(n_channels_to_use))
        
        # Calculate dimensions after pooling and channel selection
        pooled_spatial = 2 * 2 * 2  # After adaptive_avg_pool3d to (2,2,2)
        bottleneck_flat_dim = n_channels_to_use * pooled_spatial
        
        print(f"Using {n_channels_to_use}/{total_channels} channels ({self.channel_portion:.1%})")
        print(f"Flattened dimension after pooling: {bottleneck_flat_dim}")
        
        self.projector = nn.Sequential(
            nn.Linear(bottleneck_flat_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, subject_dim),
            nn.LayerNorm(subject_dim)
        )
    
    def forward(self, latent, batch_size):
        # Select portion of channels BEFORE pooling: [batch, 320, 5, 5, 5] -> [batch, n_selected, 5, 5, 5]
        if len(self.selected_channels) < latent.shape[1]:
            latent_selected = latent[:, self.selected_channels, :, :, :]
        else:
            latent_selected = latent
        
        # Apply spatial pooling to selected channels: [batch, n_selected, 5, 5, 5] -> [batch, n_selected, 2, 2, 2]
        pooled = F.adaptive_avg_pool3d(latent_selected, (2, 2, 2))
        # Flatten: [batch, n_selected * 2 * 2 * 2]
        bottleneck_flat = pooled.view(batch_size, -1)
        return self.projector(bottleneck_flat)



class TriplePooledProgressiveProjector(nn.Module):
    """Spatial pooling + progressive reduction with three-way channel splitting.
    Projects subject and uniqueness channels, returns remaining channels as raw features."""
    
    def __init__(self, bottleneck_dim, subject_dim, subject_portion=0.4, uniqueness_portion=0.4, 
                 num_unique_classes=None, use_softmax=False):
        """
        Args:
            bottleneck_dim: Original bottleneck dimensions as shape (e.g., [320, 5, 5, 5])
            subject_dim: Target embedding dimension for subject output only
            subject_portion: Portion of channels for subject_latent projection (0.0 to 0.9)
            uniqueness_portion: Portion of channels for uniqueness_latent projection (0.0 to 0.9)
            num_unique_classes: Number of unique classes for classification output.
                              If None, will output a single value for uniqueness scoring.
            use_softmax: Whether to apply softmax to uniqueness output (for probabilities)
            
        Note: subject_portion + uniqueness_portion must be <= 0.9 to ensure at least 10% 
              channels remain for raw features (no projection)
        """
        super().__init__()
        
        # Store classification parameters
        self.num_unique_classes = num_unique_classes
        self.use_softmax = use_softmax
        
        # Determine uniqueness output dimension
        if num_unique_classes is not None:
            uniqueness_output_dim = num_unique_classes
            print(f"Uniqueness projector configured for {num_unique_classes}-class classification")
        else:
            uniqueness_output_dim = 1  # Single uniqueness score
            print("Uniqueness projector configured for single uniqueness scoring")
        
        # Validate input dimensions
        if isinstance(bottleneck_dim, (list, tuple)):
            self.bottleneck_shape = bottleneck_dim
            total_channels = bottleneck_dim[0]  # 320
            self.spatial_dims = bottleneck_dim[1:]  # [5, 5, 5]
        else:
            raise ValueError("bottleneck_dim should be the shape [channels, d, h, w], not flattened dimension")
        
        # Validate portions
        self.subject_portion = max(0.0, min(0.9, subject_portion))
        self.uniqueness_portion = max(0.0, min(0.9, uniqueness_portion))
        
        # Ensure total doesn't exceed 0.9
        total_portion = self.subject_portion + self.uniqueness_portion
        if total_portion > 0.9:
            # Scale down proportionally to fit within 0.9
            scale_factor = 0.9 / total_portion
            self.subject_portion *= scale_factor
            self.uniqueness_portion *= scale_factor
            print(f"Warning: Portions scaled down to fit 0.9 limit. New portions: "
                  f"subject={self.subject_portion:.3f}, uniqueness={self.uniqueness_portion:.3f}")
        
        # Calculate number of channels for each group
        n_subject_channels = max(1, int(total_channels * self.subject_portion))
        n_uniqueness_channels = max(1, int(total_channels * self.uniqueness_portion))
        
        # Ensure we don't exceed total channels
        if n_subject_channels + n_uniqueness_channels >= total_channels:
            # Adjust to leave at least 1 channel for remaining
            available_for_first_two = total_channels - 1
            ratio = n_subject_channels / (n_subject_channels + n_uniqueness_channels)
            n_subject_channels = max(1, int(available_for_first_two * ratio))
            n_uniqueness_channels = max(1, available_for_first_two - n_subject_channels)
        
        # Create channel indices for each group
        self.subject_channels = list(range(n_subject_channels))
        self.uniqueness_channels = list(range(n_subject_channels, n_subject_channels + n_uniqueness_channels))
        #self.remaining_channels = list(range(n_subject_channels + n_uniqueness_channels, total_channels))
        
        # Calculate dimensions after pooling
        pooled_spatial = 2 * 2 * 2  # After adaptive_avg_pool3d to (2,2,2)
        subject_flat_dim = len(self.subject_channels) * pooled_spatial
        uniqueness_flat_dim = len(self.uniqueness_channels) * pooled_spatial
        #remaining_flat_dim = len(self.remaining_channels) * pooled_spatial
        
        print(f"Channel distribution:")
        print(f"  Subject channels: {len(self.subject_channels)}/{total_channels} "
              f"({len(self.subject_channels)/total_channels:.1%}) - indices {self.subject_channels[:3]}...")
        print(f"  Uniqueness channels: {len(self.uniqueness_channels)}/{total_channels} "
              f"({len(self.uniqueness_channels)/total_channels:.1%}) - indices {self.uniqueness_channels[:3]}...")
        #print(f"  Remaining channels: {len(self.remaining_channels)}/{total_channels} "
        #      f"({len(self.remaining_channels)/total_channels:.1%}) - indices {self.remaining_channels[:3]}... (no projection)")
        print(f"Flattened dimensions after pooling:")
        print(f"  Subject: {subject_flat_dim} -> projected, Uniqueness: {uniqueness_flat_dim} -> projected")
        #print(f"  Remaining: {remaining_flat_dim} -> raw features (no projection)")
        
        # Helper function to calculate first layer size based on input dimension
        def calculate_first_layer_size(input_dim):
            """Calculate first layer size using (x)^3 where x is the floor power of 2."""
            if input_dim <= 0:
                return 64  # Minimum fallback
            
            # Find the power such that 2^power <= input_dim < 2^(power+1)
            power = int(np.floor(np.log2(input_dim)))
            first_layer_size = 2**(power + 1)
            
            # Ensure minimum size and reasonable maximum
            first_layer_size = max(64, min(first_layer_size, 4096))
            
            print(f"Input dim: {input_dim} -> Power: {power} -> First layer: {first_layer_size}")
            return first_layer_size
        
        # Calculate first layer sizes for each projector (only for projected outputs)
        subject_first_layer = calculate_first_layer_size(subject_flat_dim)
        uniqueness_first_layer = calculate_first_layer_size(uniqueness_flat_dim)
        
        # Projector for subject channels
        self.subject_projector = nn.Sequential(
            nn.Linear(subject_flat_dim, subject_first_layer),
            nn.LayerNorm(subject_first_layer),
            nn.ReLU(),
            
            nn.Linear(subject_first_layer, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, subject_dim),
            nn.LayerNorm(subject_dim)
        )
        
        # Projector for uniqueness channels (classification)
        uniqueness_layers = [
            nn.Linear(uniqueness_flat_dim, uniqueness_first_layer),
            nn.LayerNorm(uniqueness_first_layer),
            nn.ReLU(),
            
            nn.Linear(uniqueness_first_layer, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            
            nn.Linear(256, uniqueness_output_dim)  # No LayerNorm for classification logits
        ]
        
        # Add softmax if requested
        if self.use_softmax and self.num_unique_classes is not None:
            uniqueness_layers.append(nn.Softmax(dim=1))
        
        self.uniqueness_projector = nn.Sequential(*uniqueness_layers)
        
        # No projector for remaining channels - will return raw features
    
    def forward(self, latent, batch_size):
        """
        Args:
            latent: Input tensor of shape [batch, channels, d, h, w]
            batch_size: Batch size
            
        Returns:
            tuple: (subject_latent, uniqueness_latent, remaining_features)
                - subject_latent: Embedding of subject channels (shape: [batch, subject_dim])
                - uniqueness_latent: Classification logits/scores for uniqueness 
                                   (shape: [batch, num_unique_classes] or [batch, 1])
                - remaining_features: Raw pooled features from remaining channels 
                                    (shape: [batch, remaining_flat_dim])
        """
        # Process subject channels
        latent_subject = latent[:, self.subject_channels, :, :, :]
        pooled_subject = F.adaptive_avg_pool3d(latent_subject, (2, 2, 2))
        flat_subject = pooled_subject.view(batch_size, -1)
        subject_latent = self.subject_projector(flat_subject)
        
        # Process uniqueness channels
        latent_uniqueness = latent[:, self.uniqueness_channels, :, :, :]
        pooled_uniqueness = F.adaptive_avg_pool3d(latent_uniqueness, (2, 2, 2))
        flat_uniqueness = pooled_uniqueness.view(batch_size, -1)
        uniqueness_latent = self.uniqueness_projector(flat_uniqueness)
        
        # Process remaining channels (no projection - return raw features)
        #latent_remaining = latent[:, self.remaining_channels, :, :, :]
        #pooled_remaining = F.adaptive_avg_pool3d(latent_remaining, (2, 2, 2))
        #remaining_features = pooled_remaining.view(batch_size, -1)  # Raw flattened features
        
        return subject_latent, uniqueness_latent#, remaining_features


# Alternative version with configurable architecture for each projector
class TriplePooledProgressiveProjectorCustom(TriplePooledProgressiveProjector):
    """Version with customizable architectures for each projector."""
    
    def __init__(self, bottleneck_dim, subject_dim, subject_portion=0.4, uniqueness_portion=0.4,
                 hidden_dims=[1024, 256], use_dropout=False, dropout_rate=0.1,
                 num_unique_classes=None, use_softmax=False):
        """
        Args:
            bottleneck_dim: Original bottleneck dimensions as shape (e.g., [320, 5, 5, 5])
            subject_dim: Target embedding dimension for subject output only
            subject_portion: Portion of channels for subject_latent projection (0.0 to 0.9)
            uniqueness_portion: Portion of channels for uniqueness_latent projection (0.0 to 0.9)
            hidden_dims: List of hidden layer dimensions
            use_dropout: Whether to add dropout layers
            dropout_rate: Dropout rate if use_dropout is True
            num_unique_classes: Number of unique classes for classification output
            use_softmax: Whether to apply softmax to uniqueness output
        """
        # Call parent's __init__ but we'll override the projectors
        super().__init__(bottleneck_dim, subject_dim, subject_portion, uniqueness_portion, 
                        num_unique_classes, use_softmax)
        
        # Calculate dimensions after pooling
        pooled_spatial = 2 * 2 * 2
        subject_flat_dim = len(self.subject_channels) * pooled_spatial
        uniqueness_flat_dim = len(self.uniqueness_channels) * pooled_spatial
        remaining_flat_dim = len(self.remaining_channels) * pooled_spatial
        
        # Helper function to calculate first layer size based on input dimension
        def calculate_first_layer_size(input_dim):
            """Calculate first layer size using (x)^3 where x is the floor power of 2."""
            if input_dim <= 0:
                return 64  # Minimum fallback
            
            # Find the power such that 2^power <= input_dim < 2^(power+1)
            power = int(np.floor(np.log2(input_dim)))
            first_layer_size = power ** 3
            
            # Ensure minimum size and reasonable maximum
            first_layer_size = max(64, min(first_layer_size, 4096))
            
            print(f"Custom - Input dim: {input_dim} -> Power: {power} -> First layer: {first_layer_size}")
            return first_layer_size
        
        # Helper function to create projector
        def create_projector(input_dim):
            layers = []
            prev_dim = input_dim
            
            # Use dynamic first layer size instead of first element in hidden_dims
            first_layer_size = calculate_first_layer_size(input_dim)
            layers.extend([
                nn.Linear(prev_dim, first_layer_size),
                nn.LayerNorm(first_layer_size),
                nn.ReLU()
            ])
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = first_layer_size
            
            # Continue with the rest of hidden_dims
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                ])
                if use_dropout:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            # Final layer
            layers.extend([
                nn.Linear(prev_dim, subject_dim),
                nn.LayerNorm(subject_dim)
            ])
            
            return nn.Sequential(*layers)
        
        # Helper function to create projector
        def create_projector(input_dim, output_dim, is_classification=False):
            layers = []
            prev_dim = input_dim
            
            # Use dynamic first layer size instead of first element in hidden_dims
            first_layer_size = calculate_first_layer_size(input_dim)
            layers.extend([
                nn.Linear(prev_dim, first_layer_size),
                nn.LayerNorm(first_layer_size),
                nn.ReLU()
            ])
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = first_layer_size
            
            # Continue with the rest of hidden_dims
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU()
                ])
                if use_dropout:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            
            # Final layer
            layers.append(nn.Linear(prev_dim, output_dim))
            
            # Only add LayerNorm for embeddings, not for classification
            if not is_classification:
                layers.append(nn.LayerNorm(output_dim))
            elif self.use_softmax and self.num_unique_classes is not None:
                layers.append(nn.Softmax(dim=1))
            
            return nn.Sequential(*layers)
        
        # Determine uniqueness output dimension
        if self.num_unique_classes is not None:
            uniqueness_output_dim = self.num_unique_classes
        else:
            uniqueness_output_dim = 1
        
        # Override projectors with custom architecture (no remaining projector)
        self.subject_projector = create_projector(subject_flat_dim, subject_dim, is_classification=False)
        self.uniqueness_projector = create_projector(uniqueness_flat_dim, uniqueness_output_dim, is_classification=True)
        # No remaining projector - will return raw features


class BottleNeckContrastiveLoss(nn.Module):
    def __init__(self, feat_weight: float = 0.1, in_dim=1, out_dim=1):
        super().__init__()
        self.feat_weight = feat_weight
        self.proj_latent = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, batch, output, mask, latent):        
        #print(len(latent)) # why is this a list??
        #print(latent[0].shape)
        return torch.mean(latent[0]) + torch.mean(output**2) 


class SubjectImageSimilarityLoss(nn.Module):
    def __init__(self, bottleneck_dim, subject_dim,
                 similarity_weight=1.0, variance_weight=0.1,
                 image_projector=None, channels_proportion_at_embedding = 1.0):  # Pass module object here
        super().__init__()
        self.similarity_weight = similarity_weight
        self.variance_weight = variance_weight
        
        # Use provided projector or default to ProgressiveProjector
        if image_projector is None:
            self.image_to_subject_projector = PooledProgressiveProjector(bottleneck_dim, subject_dim, channel_portion=channels_proportion_at_embedding)
        else:
            self.image_to_subject_projector = image_projector
        
        # NO learnable projection for subjects!
        
    def normalize_subjects(self, subject_features):
        """Simple, deterministic normalization - preserves structure"""
        # Option 1: Just L2 normalization
        return F.normalize(subject_features, dim=1, eps=1e-8)
        
       
    def extract_subject_data(self, batch, device='cuda'):
        """Extract subject features and IDs from batch dictionary."""

        subject_features = []
        
        for i in range(len(batch['properties'])):
            case_values = list(batch['properties'][i]['subject_features'].values())
            subject_features.append(case_values)
        
        subject_ids = batch.get('subject_ids', None) # TODO
        
        if subject_features is None:
            raise KeyError("'subject_features' not found in batch")

        subject_features = torch.tensor(subject_features, dtype=torch.float32)
        subject_features = subject_features.to(device)
        # CRITICAL FIX: Row-wise normalization (per case, independent of batch)
        #row_means = subject_features.mean(dim=1, keepdim=True)
        #row_stds = subject_features.std(dim=1, keepdim=True, unbiased=False)
        
        # Avoid division by zero: if std is too small, don't normalize that row
        #safe_std = torch.where(row_stds < 1e-6, torch.ones_like(row_stds), row_stds)
        #subject_features = (subject_features - row_means) / safe_std
        
        # Clip extreme values after normalization
        #subject_features = torch.clamp(subject_features, min=-10.0, max=10.0)
            
        return subject_features, subject_ids
    
    def compute_similarity_loss(self, image_emb, subject_emb):
        """
        Compute direct cosine similarity between corresponding pairs.
        High similarity = low loss.
        """
        # Add safety checks before normalization
        if torch.isnan(image_emb).any() or torch.isnan(subject_emb).any():
            print("WARNING: NaN in embeddings before similarity")
            return torch.tensor(0.0, device=image_emb.device), torch.zeros(image_emb.shape[0], device=image_emb.device)
        
        # Row-wise L2 normalization with safety for zero vectors
        def safe_normalize(x):
            norms = torch.norm(x, dim=1, keepdim=True)
            safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
            return x / safe_norms
        
        # Normalize embeddings (essential for avoiding trivial solutions)
        image_emb = safe_normalize(image_emb)#F.normalize(image_emb, dim=1, eps=1e-8)
        subject_emb = safe_normalize(subject_emb)#F.normalize(subject_emb, dim=1, eps=1e-8)
        
        # Compute cosine similarity for each pair
        cosine_similarities = F.cosine_similarity(image_emb, subject_emb, dim=1)
        
        # We want high similarity, so minimize negative similarity
        similarity_loss = -cosine_similarities.mean()
        
        return similarity_loss, cosine_similarities
    
    def compute_variance_regularization(self, image_emb, subject_emb):
        """
        Encourage embeddings to have sufficient spread using standard deviation.
        Most numerically stable approach.
        """
        eps = 1e-6
        
        # Compute standard deviation across batch for each dimension, then average
        image_std = torch.std(image_emb, dim=0, unbiased=False).mean()
        #subject_std = torch.std(subject_emb, dim=0, unbiased=False).mean()
        
        # Encourage std to be at least 0.1 (adjustable target)
        target_std = 0.1
        
        # Only penalize when std is below target (using ReLU)
        image_penalty = torch.clamp(target_std - image_std, min=0, max=100)
        #subject_penalty = torch.relu(target_std - subject_std)
        
        return image_penalty #+ subject_penalty
    
    def compute_covariance_regularization(self, image_emb, subject_emb):
        """
        Decorrelate features - normalized version for numerical stability.
        """
        def off_diagonal_covariance(x):
            if x.size(0) <= 1:  # Skip if batch too small
                return torch.tensor(0.0, device=x.device)
            
            # Center the data
            x_centered = x - x.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = torch.mm(x_centered.T, x_centered) / (x.size(0) - 1)
            
            # Get off-diagonal elements using mask
            mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
            off_diag_elements = cov[mask]
            
            # Return mean of squared off-diagonal elements (normalized!)
            return torch.mean(off_diag_elements**2)
        
        image_cov_loss = off_diagonal_covariance(image_emb)
        subject_cov_loss = off_diagonal_covariance(subject_emb)
    
        return image_cov_loss + subject_cov_loss
    
    def compute_embedding_stats(self, image_emb, subject_emb):
        """Check if embeddings are actually diverse"""
        # Compute pairwise distances within each modality
        image_distances = torch.pdist(image_emb).mean()
        subject_distances = torch.pdist(subject_emb).mean()
        
        # Compute standard deviation across the batch
        image_std = torch.std(image_emb, dim=0).mean()
        subject_std = torch.std(subject_emb, dim=0).mean()
        
        return {
            'image_pairwise_dist': image_distances.item(),
            'subject_pairwise_dist': subject_distances.item(), 
            'image_std': image_std.item(),
            'subject_std': subject_std.item()
        }

    
    def forward(self, batch, output, mask, latent):
        subject_features, _ = self.extract_subject_data(batch, output.device)


        # Add safety checks for each row
        if torch.isnan(subject_features).any():
            print("WARNING: NaN detected in subject_features")
            subject_features = torch.nan_to_num(subject_features, nan=0.0)
        
        # Check for rows with extreme values and handle them individually
        #row_maxes = subject_features.abs().max(dim=1, keepdim=True)[0]
        #extreme_rows = row_maxes > 1e6
        #if extreme_rows.any():
        #    print(f"WARNING: {extreme_rows.sum()} rows with extreme values")
            # Clip extreme rows individually
        #    subject_features = torch.where(
        #        extreme_rows.expand_as(subject_features),
        #        torch.clamp(subject_features, min=-1e6, max=1e6),
        #        subject_features
        #    )
        
        latent = latent[0]
        batch_size = latent.shape[0]
        
        # Project images to subject space
        image_projected = self.image_to_subject_projector(latent, batch_size)


        # Check for NaN after projection
        if torch.isnan(image_projected).any():
            print("WARNING: NaN detected in image_projected")
            return torch.tensor(0.0, device=output.device, requires_grad=True)
        
        # Subjects: just normalize (preserve original meaning!)
        subject_normalized = self.normalize_subjects(subject_features)
        
        # Direct comparison in subject space
        similarity_loss, cosine_sims = self.compute_similarity_loss(image_projected, subject_normalized)
        
        # Only regularize the learnable part (image projections)
        #variance_loss = self.compute_variance_regularization(image_projected, image_projected)
        
        total_loss = self.similarity_weight * similarity_loss #+ self.variance_weight * variance_loss)
        #print(f"Similarity Loss: {similarity_loss.item()}, Variance Loss: {variance_loss.item()}")
        #stats = self.compute_embedding_stats(image_projected, subject_normalized)
        #print(f"Embedding Stats: {stats}")
        
        return total_loss


class AlternativeVICRegLoss(nn.Module):
    """
    Alternative approach using VICReg-style loss (Variance-Invariance-Covariance).
    Very robust against collapse.
    """
    def __init__(self, bottleneck_dim, subject_dim, projection_dim=256, 
                 sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight 
        self.cov_weight = cov_weight
        
        self.image_projector = nn.Sequential(
            nn.Linear(bottleneck_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.subject_projector = nn.Sequential(
            nn.Linear(subject_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def extract_subject_data(self, batch):
        subject_features = batch.get('subject_features', None)
        subject_ids = batch.get('subject_ids', None)
        
        if subject_features is None or subject_ids is None:
            raise KeyError("Missing subject data in batch")
            
        return subject_features, subject_ids
    
    def invariance_loss(self, z1, z2):
        """Minimize distance between corresponding pairs."""
        return F.mse_loss(z1, z2)
    
    def variance_loss(self, z):
        """Maintain variance above threshold."""
        std_z = torch.sqrt(z.var(dim=0) + 1e-04)
        return torch.mean(F.relu(1 - std_z))
    
    def covariance_loss(self, z):
        """Decorrelate features."""
        batch_size, dim = z.shape
        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / (batch_size - 1)
        
        # Off-diagonal elements should be close to 0
        off_diag_cov = cov_z.flatten()[:-1].view(dim-1, dim+1)[:, 1:].flatten()
        return off_diag_cov.pow(2).sum() / dim
    
    def forward(self, batch, output, mask, latent):
        subject_features, subject_ids = self.extract_subject_data(batch)
        
        batch_size = latent.size(0)
        bottleneck_flat = latent.view(batch_size, -1)
        
        # Project both modalities
        z_image = self.image_projector(bottleneck_flat)
        z_subject = self.subject_projector(subject_features)
        
        # VICReg loss components
        inv_loss = self.invariance_loss(z_image, z_subject)
        var_loss = self.variance_loss(z_image) + self.variance_loss(z_subject)
        cov_loss = self.covariance_loss(z_image) + self.covariance_loss(z_subject)
        
        total_loss = (self.sim_weight * inv_loss + 
                     self.var_weight * var_loss + 
                     self.cov_weight * cov_loss)
        
        # Store for monitoring
        self.last_inv_loss = inv_loss.item()
        self.last_var_loss = var_loss.item()
        self.last_cov_loss = cov_loss.item()
        
        return total_loss


class ReconstructionAndSimilarityLoss(nn.Module):
    def __init__(self, 
                 # Similarity loss parameters
                 bottleneck_dim, 
                 subject_dim,
                 similarity_kwargs=None,
                 # Loss weights
                 weight_reconstruction=1.0, 
                 weight_similarity=0.1):
        """
        Compound loss combining reconstruction (MSE) and subject-image similarity.
        
        Args:
            bottleneck_dim: Dimensions of the latent bottleneck
            subject_dim: Dimension of subject features
            similarity_kwargs: Dict of kwargs for SubjectImageSimilarityLoss
            weight_reconstruction: Weight for reconstruction loss
            weight_similarity: Weight for similarity loss
        """
        super().__init__()
        
        self.weight_reconstruction = weight_reconstruction
        self.weight_similarity = weight_similarity
        
        # Initialize the component losses
        self.reconstruction_loss = MAEMSELoss()
        
        # Default similarity loss parameters
        if similarity_kwargs is None:
            similarity_kwargs = {
                'similarity_weight': 1.0,
                'variance_weight': 0.1
            }
        
        self.similarity_loss = SubjectImageSimilarityLoss(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            **similarity_kwargs
        )
    
    def forward(self, batch, model_output, target, mask, latent):
        """
        Forward pass computing both reconstruction and similarity losses.
        
        Args:
            batch: Batch dictionary containing subject features
            model_output: Model's reconstruction output
            target: Ground truth target
            loss_mask: Mask for reconstruction loss calculation
            latent: Latent representations from the model
            
        Returns:
            Combined weighted loss
        """
        
        # Compute reconstruction loss (original task)
        recon_loss = self.reconstruction_loss.forward(
            model_output=model_output,
            target=target, 
            mask=mask
        ) if self.weight_reconstruction != 0 else 0
        
        # Compute similarity loss (new contrastive task)
        sim_loss = self.similarity_loss.forward(
            batch=batch,
            output=model_output,  # Pass model_output as 'output'
            mask=mask,
            latent=latent
        ) if self.weight_similarity != 0 else 0
        
        # Combine losses with weights
        total_loss = (self.weight_reconstruction * recon_loss + 
                     self.weight_similarity * sim_loss)
        
        # Optional: Print component losses for monitoring
        if self.weight_reconstruction != 0 and self.weight_similarity != 0:
            logger.info(f"Recon Loss: {recon_loss:.6f}, Sim Loss: {sim_loss:.6f}, Total: {total_loss:.6f} with {len(self.similarity_loss.image_to_subject_projector.selected_channels)} bottleneck channels")
        
        return total_loss

class ReconstructionAndSimilarityLossPortion05(ReconstructionAndSimilarityLoss):
    """
    Inherits from ReconstructionAndSimilarityLoss but uses only 50% of channels 
    for the similarity loss computation.
    """
    
    def __init__(self,
                 # Similarity loss parameters
                 bottleneck_dim, 
                 subject_dim,
                 similarity_kwargs=None,
                 # Loss weights
                 weight_reconstruction=1.0, 
                 weight_similarity=0.1):
        """
        Initialize with same parameters as parent, but automatically set 
        channels_proportion_at_embedding to 0.5
        
        Args:
            bottleneck_dim: Dimensions of the latent bottleneck 
            subject_dim: Dimension of subject features
            similarity_kwargs: Dict of kwargs for SubjectImageSimilarityLoss
            weight_reconstruction: Weight for reconstruction loss
            weight_similarity: Weight for similarity loss
        """
        
        # Default similarity loss parameters
        if similarity_kwargs is None:
            similarity_kwargs = {
                'similarity_weight': 1.0,
                'variance_weight': 0.1
            }
        
        # Add the 50% channel portion parameter
        similarity_kwargs['channels_proportion_at_embedding'] = 0.5
        
        # Call parent constructor with modified similarity_kwargs
        super().__init__(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            similarity_kwargs=similarity_kwargs,
            weight_reconstruction=weight_reconstruction,
            weight_similarity=weight_similarity
        )


class TripleSubjectSimilarityLoss(nn.Module):
    def __init__(self, bottleneck_dim, subject_dim,
                 similarity_weight=1.0, classification_weight=1.0, variance_weight=0.1,
                 subject_portion=0.5, uniqueness_portion=0.3, num_unique_classes=None,
                 image_projector=None, use_softmax=False):
        """
        Triple projector loss combining subject similarity and uniqueness classification.
        
        Args:
            bottleneck_dim: Input dimensions [channels, d, h, w]
            subject_dim: Subject embedding dimension
            similarity_weight: Weight for subject similarity loss
            classification_weight: Weight for uniqueness classification loss
            variance_weight: Weight for variance regularization
            subject_portion: Portion of channels for subject embedding
            uniqueness_portion: Portion of channels for uniqueness classification
            num_unique_classes: Number of unique classes for classification
            image_projector: Custom projector module (if None, creates TriplePooledProgressiveProjector)
            use_softmax: Whether to apply softmax to uniqueness output
        """
        super().__init__()
        self.similarity_weight = similarity_weight
        self.classification_weight = classification_weight
        self.variance_weight = variance_weight
        
        # Use provided projector or create default TriplePooledProgressiveProjector
        if image_projector is None:
            self.image_to_subject_projector = TriplePooledProgressiveProjector(
                bottleneck_dim=bottleneck_dim,
                subject_dim=subject_dim,
                subject_portion=subject_portion,
                uniqueness_portion=uniqueness_portion,
                num_unique_classes=num_unique_classes,
                use_softmax=use_softmax
            )
        else:
            self.image_to_subject_projector = image_projector
            
        # Classification loss criterion
        self.classification_criterion = nn.CrossEntropyLoss()
        
    def normalize_subjects(self, subject_features):
        """Simple, deterministic normalization - preserves structure"""
        return F.normalize(subject_features, dim=1, eps=1e-8)
        
    def extract_subject_data(self, batch, device='cuda'):
        """Extract subject features and IDs from batch dictionary."""
        subject_features = []
        
        for i in range(len(batch['properties'])):
            case_values = list(batch['properties'][i]['subject_features'].values())
            subject_features.append(case_values)
        
        subject_ids = batch.get('subject_ids', None)
        
        if subject_features is None:
            raise KeyError("'subject_features' not found in batch")

        subject_features = torch.tensor(subject_features, dtype=torch.float32)
        subject_features = subject_features.to(device)
            
        return subject_features, subject_ids
    
    def extract_uniqueness_labels(self, batch, device='cuda'):
        """Extract uniqueness classification labels from batch."""
        uniqueness_labels = []
        for i in range(len(batch['properties'])):
            label = 0.0 if batch['properties'][i]['extra_info']['subject_info']['health_status'] == 'healthy' else 1.0
            uniqueness_labels.append(label)
        uniqueness_labels = torch.tensor(uniqueness_labels, dtype=torch.long)
        
        return uniqueness_labels.to(device)
    
    def compute_similarity_loss(self, image_emb, subject_emb):
        """
        Compute direct cosine similarity between corresponding pairs.
        High similarity = low loss.
        """
        # Add safety checks before normalization
        if torch.isnan(image_emb).any() or torch.isnan(subject_emb).any():
            print("WARNING: NaN in embeddings before similarity")
            return torch.tensor(0.0, device=image_emb.device), torch.zeros(image_emb.shape[0], device=image_emb.device)
        
        # Row-wise L2 normalization with safety for zero vectors
        def safe_normalize(x):
            norms = torch.norm(x, dim=1, keepdim=True)
            safe_norms = torch.where(norms < 1e-8, torch.ones_like(norms), norms)
            return x / safe_norms
        
        # Normalize embeddings (essential for avoiding trivial solutions)
        #image_emb = safe_normalize(image_emb)
        #subject_emb = safe_normalize(subject_emb)
        
        # Compute cosine similarity for each pair
        cosine_similarities = F.cosine_similarity(image_emb, subject_emb, dim=1)
        
        # We want high similarity, so minimize negative similarity
        similarity_loss = -cosine_similarities.mean()
        
        return similarity_loss, cosine_similarities
    
    def compute_variance_regularization(self, image_emb):
        """
        Encourage embeddings to have sufficient spread using standard deviation.
        Most numerically stable approach.
        """
        eps = 1e-6
        
        # Compute standard deviation across batch for each dimension, then average
        image_std = torch.std(image_emb, dim=0, unbiased=False).mean()
        
        # Encourage std to be at least 0.1 (adjustable target)
        target_std = 0.1
        
        # Only penalize when std is below target (using ReLU)
        image_penalty = torch.clamp(target_std - image_std, min=0, max=100)
        
        return image_penalty
    


    def forward(self, batch, output, mask, latent):
        """
        Forward pass computing both similarity and classification losses.
        
        Returns:
            dict: Dictionary containing total loss and individual components
        """
        # Extract subject features and uniqueness labels
        subject_features, _ = self.extract_subject_data(batch, output.device)
        uniqueness_labels = self.extract_uniqueness_labels(batch, output.device)

        # Add safety checks for subject features
        if torch.isnan(subject_features).any():
            print("WARNING: NaN detected in subject_features")
            subject_features = torch.nan_to_num(subject_features, nan=0.0)
        
        # Prepare latent input
        latent = latent[0]
        batch_size = latent.shape[0]
        
        # Project images using triple projector
        subject_projected, uniqueness_logits = self.image_to_subject_projector(latent, batch_size)

        # Check for NaN after projection
        if torch.isnan(subject_projected).any():
            print("WARNING: NaN detected in subject_projected")
            return {
                'total_loss': torch.tensor(0.0, device=output.device, requires_grad=True),
                'similarity_loss': torch.tensor(0.0, device=output.device),
                'classification_loss': torch.tensor(0.0, device=output.device),
                'classification_accuracy': torch.tensor(0.0, device=output.device)
            }
        
        # ===== SUBJECT SIMILARITY LOSS =====
        # Normalize subject features (preserve original meaning!)
        #subject_normalized = self.normalize_subjects(subject_features)
        
        # Compute similarity loss between projected images and subject features
        similarity_loss, cosine_sims = self.compute_similarity_loss(subject_projected, subject_features)
        
        # ===== UNIQUENESS CLASSIFICATION LOSS =====
        if torch.isnan(uniqueness_logits).any():
            print("WARNING: NaN detected in uniqueness_logits")
            classification_loss = torch.tensor(0.0, device=output.device, requires_grad=True)
            classification_accuracy = torch.tensor(0.0, device=output.device)
        else:
            classification_loss = self.classification_criterion(uniqueness_logits, uniqueness_labels)
            
            # Compute classification accuracy
            with torch.no_grad():
                pred_classes = torch.argmax(uniqueness_logits, dim=1)
                classification_accuracy = (pred_classes == uniqueness_labels).float().mean()
        
        # ===== VARIANCE REGULARIZATION =====
        #variance_loss = self.compute_variance_regularization(subject_projected)
        
        # ===== COMBINE LOSSES =====
        total_loss = (self.similarity_weight * similarity_loss + 
                     self.classification_weight * classification_loss) #+ self.variance_weight * variance_loss)
        
        # ===== OPTIONAL: EMBEDDING STATS =====
        #if hasattr(self, 'debug') and self.debug:
        #    stats = self.compute_embedding_stats(subject_projected, subject_normalized)
        #    print(f"Embedding Stats: {stats}")
        #    print(f"Similarity Loss: {similarity_loss.item():.4f}, "
        #          f"Classification Loss: {classification_loss.item():.4f}, "
        #          f"Variance Loss: {variance_loss.item():.4f}")
        
        return {
            'total_loss': total_loss,
            'similarity_loss': similarity_loss,
            'classification_loss': classification_loss,
            'classification_accuracy': classification_accuracy,
            #'variance_loss': variance_loss,
            'cosine_similarities': cosine_sims,
            'subject_embeddings': subject_projected,
            'uniqueness_logits': uniqueness_logits
            #'raw_features': raw_features
        }


# Alternative simplified version that returns just the total loss (like original)
class TripleSubjectSimilarityLossSimple(TripleSubjectSimilarityLoss):
    """Simplified version that returns just the total loss for backward compatibility."""
    
    def forward(self, batch, output, mask, latent):
        result = super().forward(batch, output, mask, latent)
        return result['total_loss']


# Example usage function
def create_triple_loss_module(bottleneck_dim, subject_dim, num_unique_classes,
                             similarity_weight=1.0, classification_weight=1.0):
    """
    Helper function to create a properly configured TripleSubjectSimilarityLoss.
    
    Args:
        bottleneck_dim: Input dimensions [channels, d, h, w]
        subject_dim: Subject embedding dimension
        num_unique_classes: Number of unique subjects to classify
        similarity_weight: Weight for subject similarity loss
        classification_weight: Weight for uniqueness classification loss
    
    Returns:
        TripleSubjectSimilarityLoss instance
    """
    return TripleSubjectSimilarityLoss(
        bottleneck_dim=bottleneck_dim,
        subject_dim=subject_dim,
        similarity_weight=similarity_weight,
        classification_weight=classification_weight,
        variance_weight=0.1,
        subject_portion=0.4,
        uniqueness_portion=0.4,
        num_unique_classes=num_unique_classes,
        use_softmax=False  # Use raw logits for CrossEntropyLoss
    )



class ReconstructionAndSimilarityUniquenessLoss(nn.Module):
    def __init__(self, 
                 # Similarity loss parameters
                 bottleneck_dim, 
                 subject_dim,
                 num_unique_classes,
                 # Triple similarity loss parameters
                 triple_similarity_kwargs=None,
                 # Loss weights
                 weight_reconstruction=0.7, 
                 weight_similarity=0.15,
                 weight_classification=0.15):
        """
        Compound loss combining reconstruction (MSE) and triple subject-image similarity + classification.
        
        Args:
            bottleneck_dim: Dimensions of the latent bottleneck
            subject_dim: Dimension of subject features
            num_unique_classes: Number of unique classes for classification
            triple_similarity_kwargs: Dict of kwargs for TripleSubjectSimilarityLoss
            weight_reconstruction: Weight for reconstruction loss
            weight_similarity: Weight for subject similarity loss
            weight_classification: Weight for uniqueness classification loss
        """
        super().__init__()
        
        self.weight_reconstruction = weight_reconstruction
        self.weight_similarity = weight_similarity
        self.weight_classification = weight_classification
        
        # Initialize the component losses
        self.reconstruction_loss = MAEMSELoss()
        
        # Default triple similarity loss parameters
        if triple_similarity_kwargs is None:
            triple_similarity_kwargs = {
                'similarity_weight': 1.0,
                'classification_weight': 1.0,
                'variance_weight': 0.1,
                'subject_portion': 0.5,
                'uniqueness_portion': 0.3,
                'use_softmax': False
            }
        
        self.triple_similarity_loss = TripleSubjectSimilarityLoss(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            num_unique_classes=num_unique_classes,
            **triple_similarity_kwargs
        )
    
    def forward(self, batch, model_output, target, mask, latent):
        """
        Forward pass computing reconstruction, similarity, and classification losses.
        
        Args:
            batch: Batch dictionary containing subject features and uniqueness labels
            model_output: Model's reconstruction output
            target: Ground truth target
            mask: Mask for reconstruction loss calculation
            latent: Latent representations from the model
            
        Returns:
            dict: Dictionary containing total loss and individual components
        """
        
        # ===== RECONSTRUCTION LOSS =====
        recon_loss = 0
        if self.weight_reconstruction != 0:
            recon_loss = self.reconstruction_loss.forward(
                model_output=model_output,
                target=target, 
                mask=mask
            )
        
        # ===== TRIPLE SIMILARITY LOSS =====
        sim_results = {'total_loss': 0, 'similarity_loss': 0, 'classification_loss': 0, 'classification_accuracy': 0}
        if self.weight_similarity != 0 or self.weight_classification != 0:
            sim_results = self.triple_similarity_loss.forward(
                batch=batch,
                output=model_output,
                mask=mask,
                latent=latent
            )
        
        # Extract individual components
        similarity_loss = sim_results['similarity_loss']
        classification_loss = sim_results['classification_loss']
        classification_accuracy = sim_results['classification_accuracy']
        
        # ===== COMBINE LOSSES WITH WEIGHTS =====
        total_loss = (self.weight_reconstruction * recon_loss + 
                     self.weight_similarity * similarity_loss +
                     self.weight_classification * classification_loss)
        

        
        all_losses = {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'similarity_loss': similarity_loss,
            'classification_loss': classification_loss,
            'classification_accuracy': classification_accuracy,
            'subject_embeddings': sim_results.get('subject_embeddings', None),
            'uniqueness_logits': sim_results.get('uniqueness_logits', None),
            'raw_features': sim_results.get('raw_features', None)
        }
        logger.info(f"All Losses - Total: {total_loss:.2f}, reconstruction: {recon_loss:.2f},similarity: {similarity_loss:.2f}, classification: {classification_loss:.2f}, accuracy: {classification_accuracy:.2f}")
        return all_losses['total_loss']


class ReconstructionAndTripleSimilarityLossSimple(ReconstructionAndSimilarityUniquenessLoss):
    """Simplified version that returns just the total loss for backward compatibility."""
    
    def forward(self, batch, model_output, target, mask, latent):
        result = super().forward(batch, model_output, target, mask, latent)
        return result['total_loss']


# Helper function to create the loss module with sensible defaults
def create_reconstruction_triple_loss(bottleneck_dim, subject_dim, num_unique_classes,
                                    weight_reconstruction=1.0, weight_similarity=0.1, 
                                    weight_classification=0.1, subject_portion=0.4, 
                                    uniqueness_portion=0.4):
    """
    Helper function to create a properly configured ReconstructionAndTripleSimilarityLoss.
    
    Args:
        bottleneck_dim: Input dimensions [channels, d, h, w]
        subject_dim: Subject embedding dimension
        num_unique_classes: Number of unique subjects to classify
        weight_reconstruction: Weight for reconstruction loss
        weight_similarity: Weight for subject similarity loss
        weight_classification: Weight for uniqueness classification loss
        subject_portion: Portion of channels for subject embedding
        uniqueness_portion: Portion of channels for uniqueness classification
    
    Returns:
        ReconstructionAndTripleSimilarityLoss instance
    """
    triple_similarity_kwargs = {
        'similarity_weight': 1.0,
        'classification_weight': 1.0,
        'variance_weight': 0.1,
        'subject_portion': subject_portion,
        'uniqueness_portion': uniqueness_portion,
        'use_softmax': False
    }
    
    return ReconstructionAndSimilarityUniquenessLoss(
        bottleneck_dim=bottleneck_dim,
        subject_dim=subject_dim,
        num_unique_classes=num_unique_classes,
        triple_similarity_kwargs=triple_similarity_kwargs,
        weight_reconstruction=weight_reconstruction,
        weight_similarity=weight_similarity,
        weight_classification=weight_classification
    )


# Alternative version with separate control over inner loss weights
class ReconstructionAndSimilarityUniquenessLossAdvanced(nn.Module):
    def __init__(self, 
                 bottleneck_dim, 
                 subject_dim,
                 num_unique_classes,
                 # Outer weights (how much each loss type contributes to total)
                 weight_reconstruction=1.0,
                 weight_triple_loss=0.1,
                 # Inner weights (within the triple loss)
                 inner_similarity_weight=1.0,
                 inner_classification_weight=1.0,
                 inner_variance_weight=0.1,
                 # Channel allocation
                 subject_portion=0.4,
                 uniqueness_portion=0.4,
                 use_softmax=False):
        """
        Advanced version with separate control over inner and outer loss weights.
        
        Args:
            weight_reconstruction: Outer weight for reconstruction loss
            weight_triple_loss: Outer weight for entire triple loss
            inner_similarity_weight: Inner weight for similarity within triple loss
            inner_classification_weight: Inner weight for classification within triple loss
            inner_variance_weight: Inner weight for variance regularization
        """
        super().__init__()
        
        self.weight_reconstruction = weight_reconstruction
        self.weight_triple_loss = weight_triple_loss
        
        # Initialize component losses
        self.reconstruction_loss = MAEMSELoss()
        
        self.triple_similarity_loss = TripleSubjectSimilarityLoss(
            bottleneck_dim=bottleneck_dim,
            subject_dim=subject_dim,
            num_unique_classes=num_unique_classes,
            similarity_weight=inner_similarity_weight,
            classification_weight=inner_classification_weight,
            variance_weight=inner_variance_weight,
            subject_portion=subject_portion,
            uniqueness_portion=uniqueness_portion,
            use_softmax=use_softmax
        )
    
    def forward(self, batch, model_output, target, mask, latent):
        """Forward pass with hierarchical loss weighting."""
        
        # Reconstruction loss
        recon_loss = 0
        if self.weight_reconstruction != 0:
            recon_loss = self.reconstruction_loss.forward(
                model_output=model_output,
                target=target, 
                mask=mask
            )
        
        # Triple loss (already weighted internally)
        triple_results = {'total_loss': 0, 'similarity_loss': 0, 'classification_loss': 0, 'classification_accuracy': 0}
        if self.weight_triple_loss != 0:
            triple_results = self.triple_similarity_loss.forward(
                batch=batch,
                output=model_output,
                mask=mask,
                latent=latent
            )
        
        # Combine with outer weights
        total_loss = (self.weight_reconstruction * recon_loss + 
                     self.weight_triple_loss * triple_results['total_loss'])
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'triple_loss': triple_results['total_loss'],
            'similarity_loss': triple_results['similarity_loss'],
            'classification_loss': triple_results['classification_loss'],
            'classification_accuracy': triple_results['classification_accuracy'],
            'subject_embeddings': triple_results.get('subject_embeddings', None),
            'uniqueness_logits': triple_results.get('uniqueness_logits', None),
            'raw_features': triple_results.get('raw_features', None)
        }
