# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.
## DESIGN STEPS
### STEP 1: 
Define the Denoising Autoencoder model with convolution layers for encoding and transposed convolution layers for decoding the image.



### STEP 2: 
Initialize the model, loss function (MSELoss), and optimizer (Adam) for training.




### STEP 3: 

Load images from the dataset and add noise to the images to create noisy inputs.



### STEP 4: 
Pass the noisy images through the autoencoder to reconstruct clean images.




### STEP 5: 

Compute the loss between reconstructed and original images, then perform backpropagation and update model weights.



### STEP 6: 
Repeat the process for several epochs and visualize the denoised output images after training.





## PROGRAM

### Name:THIRUMURUGAN R

### Register Number:212223220118

```python
# Autoencoder Definition
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name:franklin raj g                   ")
    print("Register Number:   212223230058               ")

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
# Visualization function
train(model, train_loader, criterion, optimizer, epochs=5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary
<a href="https://ibb.co/gL1R1tS7"><img src="https://i.ibb.co/tTfJfYqh/Screenshot-2026-03-15-225013.png" alt="Screenshot-2026-03-15-225013" border="0"></a>


### Training loss
<a href="https://ibb.co/ty3DLmN"><img src="https://i.ibb.co/FGnzHxf/Screenshot-2026-03-15-225047.png" alt="Screenshot-2026-03-15-225047" border="0"></a>

## Original vs Noisy Vs Reconstructed Image
<a href="https://ibb.co/FL7gBG6m"><img src="https://i.ibb.co/1t289HZ7/Screenshot-2026-03-15-225023.png" alt="Screenshot-2026-03-15-225023" border="0"></a>
## RESULT
Thus,To develop a convolutional autoencoder for image denoising application,has been done by using pytorch.
