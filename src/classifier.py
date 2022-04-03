import os
import torch
import pathlib
import cv2
import copy
import time
import math
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import Resampling
from collections import OrderedDict
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from albumentations.pytorch import ToTensorV2

# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = image_filepath.split('\\')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label

def visualize_augmentations(dataset, train_image_paths, idx_to_class, idx=0, samples=10, cols=5, random_img = False):
    dataset = copy.deepcopy(dataset)
    # We remove the normalization and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
           
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab], fontsize=8)
    plt.tight_layout(pad=1)
    plt.show()    

def train_model(device, model, dataloaders, optimizer, dataset_sizes, num_epochs):
    model = model.cuda()
    # NLLLoss because our output is LogSoftmax
    criterion = nn.NLLLoss()
    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            num_batches = math.ceil(dataset_sizes[phase] / 64)

            for i, data in enumerate(dataloaders[phase], 0):
                print("Processing batch number ", i + 1, " of ", num_batches, " in epoch ", epoch)

                images = data[0].to(device)
                labels = data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Do validation on the test set
def test(model, dataloaders, device):
    model.eval()
    accuracy = 0 
    model.to(device)
    
    for i, data in enumerate(dataloaders['val'], 0):
        images = Variable(data[0])
        labels = Variable(data[1])
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
      
        print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders['val'])))

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet152()
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          #('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 39)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

def process_image(image):
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Resampling.LANCZOS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, idx_to_class, topk=5):
    image = torch.FloatTensor(np.array([process_image(Image.open(image_path))]))
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

def create_transforms():
    # Data Augmentation
    train_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
            A.RandomCrop(height=256, width=256),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ToTensorV2(),
        ]
    )
    test_transforms = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=256, width=256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    data_transforms = {
        "train": train_transforms,
        "test": test_transforms
    }
    return data_transforms

def collect_images_and_classes(classes, image_paths, subdirs):
    # Collect train
    path = os.path.join(pathlib.Path().resolve().parent, subdirs["train"])

    for root, dirs, files in os.walk(path):
        for file in files:
            classes.append(root.split('\\')[-1])
            image_paths["train"].append(os.path.join(root, file))

    # Split validation set from training data
    image_paths["train"], image_paths["val"] = image_paths["train"][:int(0.8*len(image_paths["train"]))], image_paths["train"][int(0.8*len(image_paths["train"])):] 

    # Collect test
    path = os.path.join(pathlib.Path().resolve().parent, subdirs["test"])
    
    for root, dirs, files in os.walk(path):
        for file in files:
            classes.append(root.split('\\')[-1])
            image_paths["test"].append(os.path.join(root, file))

    print("Train size: {}\nValid size: {}\nTest size: {}".format(len(image_paths["train"]), len(image_paths["val"]), len(image_paths["test"])))

def print_num_images(path):
    num_images = 0
    search_path = os.path.join(pathlib.Path().resolve().parent, path)

    for root, dirs, files in os.walk(search_path):
        for file in files:
            num_images += 1
    print("Total number of images: ", num_images)

def map_classes_to_indexes(classes):
    # Remove duplicates, but keep order
    class_types = list(dict.fromkeys(classes))
    idx_to_class = { i:j for i, j in enumerate(class_types) }
    class_to_idx = { value:key for key,value in idx_to_class.items() }
    return idx_to_class, class_to_idx

def create_checkpoint(model, class_to_idx, dataloaders, data_transforms, optimizer, num_epochs):
    model.class_to_idx = class_to_idx
    model.epochs = num_epochs
    checkpoint = {'input_size': [3, 224, 224],
                    'batch_size': dataloaders['train'].batch_size,
                    'output_size': 39,
                    'state_dict': model.state_dict(),
                    'data_transforms': data_transforms,
                    'optimizer_dict':optimizer.state_dict(),
                    'class_to_idx': model.class_to_idx,
                    'epoch': model.epochs}
    return checkpoint

def build_model():
    model = models.resnet152(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier, using ReLU activations
    # Our input_size matches the in_features of pretrained model
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(2048, 512)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(512, 39)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier
    
    return model

def main():
    ################ CONFIG
    batch_size = 64
    num_epochs = 1
    checkpoint_name = "plants_checkpoint.pth"
    data_dir = 'data\\PlantVillage'
    subdirs = {
        "train": data_dir + '\\train',
        "test": data_dir + '\\test'
    }

    ################ INIT
    if not torch.cuda.is_available():
        print("No CUDA available")
        quit()
    device = torch.device("cuda:0")  

    data_transforms = create_transforms()

    # Create sets
    classes = [] #to store class values
    image_paths = {
        "train": [],
        "val": [],
        "test": []
    }
    collect_images_and_classes(classes, image_paths, subdirs)
    
    idx_to_class, class_to_idx = map_classes_to_indexes(classes)

    # Create datasets
    train_dataset = ImageDataset(image_paths["train"], class_to_idx, data_transforms["train"])
    valid_dataset = ImageDataset(image_paths["val"], class_to_idx, data_transforms["test"])
    test_dataset = ImageDataset(image_paths["test"], class_to_idx, data_transforms["test"])
    
    # Define dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size, shuffle=True),
        'val': DataLoader(valid_dataset, batch_size, shuffle=True),
        'test': DataLoader(test_dataset, batch_size, shuffle=False)
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(valid_dataset),
        'test': len(test_dataset)
    }

    model = build_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    ################ END INIT

    ################ INFO
    print_num_images(data_dir)
    visualize_augmentations(train_dataset, image_paths["train"], idx_to_class, np.random.randint(1,len(image_paths["train"])), random_img = True)

    ################ TRAINING
    train_model(device, model, dataloaders, optimizer, dataset_sizes, num_epochs)

    ################ VALIDATION
    test(model, dataloaders, device)

    ################ SAVE
    torch.save(create_checkpoint(model, class_to_idx, dataloaders, data_transforms, optimizer, num_epochs), checkpoint_name)

    ################ LOAD
    loaded_model, class_to_idx = load_checkpoint(checkpoint_name)

    ################ PREDICT
    result = predict('C:/Users/ryzen/Greenthumb/download/septoria-leaf-spot-septoria-lycopersici-on-tomat.jpeg', loaded_model, idx_to_class)

    print("{0:.4f}".format(result[0][0]), " | ", result[1][0])
    print("{0:.4f}".format(result[0][1]), " | ", result[1][1])
    print("{0:.4f}".format(result[0][2]), " | ", result[1][2])
    print("{0:.4f}".format(result[0][3]), " | ", result[1][3])
    print("{0:.4f}".format(result[0][4]), " | ", result[1][4])

if __name__ == "__main__":
    main()