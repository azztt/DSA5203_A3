"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Ma Zhiyuan <e0983565@u.nus.edu>
"""

######### IMPORTS #########
from typing import OrderedDict
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from PIL import Image
import matplotlib.pyplot as plt
import random
###########################

########## INPUT IMAGE SIZE ##########
INPUT_SIZE = (380, 380)
######################################


########## SET RANDOM SEEDS FOR REPRODUCIBILITY ##########
RANDOM_STATE = 1234
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.random.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
##########################################################

########## DICTIONARIES FOR CLASS LABELS ##########
CLASS_LABELS = [
    'bedroom',
    'Coast',
    'Forest',
    'Highway',
    'industrial',
    'Insidecity',
    'kitchen',
    'livingroom',
    'Mountain',
    'Office',
    'OpenCountry',
    'store',
    'Street',
    'Suburb',
    'TallBuilding',
]
CLASS_TO_LABEL = {}
LABEL_TO_CLASS = {}
#################################################

########## DEVICE FOR TRAINING ##########
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device available: {}'.format(DEVICE))
#########################################


###################################### Subroutines #####################################################################


def buildClassLabelDictionaries():
    """
        Uses the class label list to generate class to label and\n
        label to class dictionary for easy conversion during processing
    """
    for i, cat in enumerate(CLASS_LABELS):
        CLASS_TO_LABEL[cat] = i+1
        LABEL_TO_CLASS[i+1] = cat

def getClassDataframe(root, imageFiles, classLabel):
    return pd.DataFrame({
        'imageFile': [os.path.join(root, file) for file in imageFiles],
        'imageLabel': [classLabel]*len(imageFiles)
    })

def collateAllData(allDataPath: str) -> pd.DataFrame:
    """
        Collect file names of all the images from all class's folders\
        and assign and store their respective class labels in a dataframe and return it

        Arguments:
            allDataPath: path to directory that contains folders for all classes of images
        
        Returns:
            A pandas dataframe with columns ```imageFile``` and ```imageLabel```
    """
    # check if the directory exists
    if not os.path.exists(allDataPath):
        raise FileNotFoundError('Given directory path does not exist !')
    
    print('Collecting all image information')
    
    # if exists, collect all image file names and assign labels and store in dataframe
    imageFilesWithLabels = pd.DataFrame({
        'imageFile': [],
        'imageLabel': []
    })

    # generate class directories
    classDirectories = [os.path.join(allDataPath, imageClass) for imageClass in CLASS_LABELS]

    # iterate over each directory and add the image info to the dataframe
    for i, classDirectory in enumerate(classDirectories):
        if not os.path.exists(classDirectory):
            print('Images for class "{}" not present. Skipping...'.format(CLASS_LABELS[i]))
            continue

        # get all image file names
        imageFiles = [file for file in os.listdir(classDirectory) if os.path.isfile(os.path.join(classDirectory, file))]

        imageFilesWithLabels = pd.concat([
            imageFilesWithLabels, getClassDataframe(classDirectory, imageFiles, i+1)
        ])
        print('Read {} files from class "{}"'.format(len(imageFiles), CLASS_LABELS[i]))
    
    return imageFilesWithLabels


def build_vocabulary(**kwargs):
    pass

def get_hist(**kwargs):
    pass

def classifier(**kwargs):
    # defining the CNN architecture
    class SceneNet(nn.Module):
        def __init__(self, numClasses=kwargs['numClasses']):
            super(SceneNet, self).__init__()
            self.cv1 = nn.Sequential(
                nn.Conv2d(1, 36, kernel_size=5, stride=2),
                nn.BatchNorm2d(36),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2) # 159x159
            )

            self.cv2 = nn.Sequential(
                nn.Conv2d(36, 96, kernel_size=5, stride=2),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size = 3, stride = 2)
            )

            self.cv3 = nn.Sequential(
                nn.Conv2d(96, 144, kernel_size=3, stride=1),
                nn.BatchNorm2d(144),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size = 3, stride = 2) # 79x79
            )
            
            self.cv4 = nn.Sequential(
                nn.Conv2d(144, 96, kernel_size=3, stride=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size = 3, stride = 2)
            )
            
            self.cv5 = nn.Sequential(
                nn.Conv2d(96, 48, kernel_size=3, stride=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 3, stride = 2) # 39x39
            )
            
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(17328, 1536),
                nn.ReLU()
            )

            self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1536, 1536),
                nn.ReLU()
            )

            # self.fc3 = nn.Sequential(
            #     nn.Dropout(0.5),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU()
            # )

            self.fc4 = nn.Sequential(
                nn.Linear(1536, numClasses),
                # nn.Softmax(1)
            )

            # for m in self.modules():
            #     if isinstance(m, nn.Conv2d):
            #         init.xavier_normal_(m.weight)
            #     # if m.bias is not None:
            #     #     init.constant_(m.bias, 0)
            #     elif isinstance(m, nn.Linear):
            #         init.xavier_normal_(m.weight)
            #         # init.constant_(m.bias, 0)

        def forward(self, x):
            # x = self.pool1(torch.relu(self.conv1(x)))
            # x = self.pool2(torch.relu(self.conv2(x)))
            # x = torch.flatten(x, 1)
            # x = torch.relu(self.fc1(x))
            # x = self.fc2(x)
            o = self.cv1(x)
            o = self.cv2(o)
            o = self.cv3(o)
            o = self.cv4(o)
            o = self.cv5(o)
            o = torch.flatten(o, 1)
            o = self.fc1(o)
            o = self.fc2(o)
            # o = self.fc3(o)
            o = self.fc4(o)
            return o
    
    model = SceneNet(numClasses=kwargs['numClasses'])

    if 'modelWeights' in kwargs:
        model.load_state_dict(kwargs['modelWeights'])
    return model

def get_accuracy(**kwargs):
    pass

def save_model(**kwargs):
    pass


########## DEFINING THE IMAGE DATA GENERATOR CLASS ##########
class ImageDataset(Dataset):
    def __init__(
            self, 
            labels, 
            imageFileLists, 
            transform=None, 
            augment=False,
            augTransform = None,
            augScale=50, 
            augClean = True
    ):
        self.imgLabels = labels.to_list()
        self.imgFiles = imageFileLists.to_list()
        self.augImgs = []
        self.augment = augment
        self.augClean = augClean
        if augment:
            print('Augmenting training data...')
            for i, imageFile in tqdm(enumerate(imageFileLists), total=len(imageFileLists)):
                root = '/'.join(os.path.split(imageFile)[:-1])
                imageName = os.path.split(imageFile)[-1]
                newFiles = []
                for j in range(augScale):
                    augImageName = 'AUG{}_{}'.format(j, imageName)
                    newImagePath = os.path.join(root, augImageName)
                    newFiles.append(newImagePath)
                    img = Image.open(imageFile)
                    augImage = augTransform(img)
                    augImage.save(newImagePath)
                self.imgLabels = self.imgLabels[:i*(augScale+1)+1]+([labels.iloc[i]]*augScale)+self.imgLabels[i*(augScale+1)+1:]
                self.imgFiles = self.imgFiles[:i*(augScale+1)+1]+newFiles+self.imgFiles[i*(augScale+1)+1:]
                self.augImgs += newFiles
            print('Augmented data by {} times'.format(augScale))
        self.transform = transform

    def __len__(self):
        return len(self.imgLabels)

    def __getitem__(self, idx):
        imgPath = self.imgFiles[idx]
        image = self.transform(read_image(imgPath))
        label = self.imgLabels[idx]
        return image, torch.tensor(label)
    
    def __del__(self):
        if self.augment and self.augClean:
            print('Deleting augmentation files...')
            for file in self.augImgs:
                os.remove(file)
            print('Done !')
#############################################################



###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""

def l2Regularize(parameters, lamb):
    return lamb*sum(torch.norm(param) for param in parameters)

def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """

    # collate the training images with labels
    allImageData = collateAllData(train_data_dir)

    # split the data into training and testing
    testPercent = 0.1
    imageRem, imageTest, labelRem, labelTest = train_test_split(
        allImageData.imageFile, 
        allImageData.imageLabel, 
        test_size=testPercent, 
        stratify=allImageData.imageLabel
    )

    # preprocessing and augmentation
    augmentationTransforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=(-20,20), translate=(0.1, 0.2)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5,5))], p=0.3),
        # transforms.Resize(size=INPUT_SIZE),
        transforms.CenterCrop(size=INPUT_SIZE),
        transforms.RandomApply([transforms.RandomCrop(size=(220, 220))], p=0.5),
    ])
    generalTransforms = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(size=INPUT_SIZE),
        transforms.CenterCrop(size=INPUT_SIZE),
        transforms.ToTensor()
    ])

    # the test data generator
    testDataset = ImageDataset(
        labels=labelTest, 
        imageFileLists=imageTest, 
        transform=generalTransforms
    )

    # split the remaining data into training and validation
    validPercent = 0.25
    imageTrain, imageValid, labelTrain, labelValid = train_test_split(
        imageRem, labelRem, test_size=validPercent, stratify=labelRem
    )

    # the validation data generator
    validDataset = ImageDataset(
        labels=labelValid, 
        imageFileLists=imageValid, 
        transform=generalTransforms
    )

    # the training data generator
    trainingDataset = ImageDataset(
        labels=labelTrain, 
        imageFileLists=imageTrain,
        transform=generalTransforms,
        # augment=True,
        # augTransform=augmentationTransforms,
        # augScale=20
    )

    epochs = 100
    trainBatchSize = 128
    validBatchSize = trainBatchSize
    lr = 0.0001
    lamb = 0.01

    model = classifier(numClasses=len(CLASS_LABELS))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # lrScheduler = optim.lr_scheduler.PolynomialLR(optimizer=optimizer, total_iters=20)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, min_lr=1e-8, factor=0.8, threshold=1e-2)

    testDataLoader = DataLoader(testDataset, batch_size=validBatchSize, shuffle=False)
    validDataLoader = DataLoader(validDataset, batch_size=validBatchSize, shuffle=False)
    trainDataLoader = DataLoader(trainingDataset, batch_size=trainBatchSize, shuffle=True)

    bestValLoss = torch.inf
    bestValAcc = 0

    trainLosses = []
    validLosses = []
    validAccs = []
    counter = 0
    for epoch in range(epochs):
        model.train()
        trainProgress = tqdm(trainDataLoader, total=len(trainDataLoader))
        trainLoss = 0
        for i, data in enumerate(trainProgress):
            images, labels = data[0], data[1]
            images = images.to(DEVICE)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels-1) + l2Regularize(model.parameters(), lamb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

            trainProgress.set_description('TRAINING Epoch [{}/{}]'.format(
                epoch+1,
                epochs
            ))
            trainProgress.set_postfix(OrderedDict(
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr']
            ))
            # lrScheduler.step()
        
        trainLosses.append(trainLoss/len(trainProgress))
        
        # Evaluate on validation data
        model.eval()  # Set the model to evaluation mode
        validLoss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            validProgress = tqdm(validDataLoader, total=len(validDataLoader))
            for i, data in enumerate(validProgress):
                images, labels = data[0], data[1]
                images = images.to(DEVICE)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels-1)
                validLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += ((predicted+1) == labels).sum().item()
                del data, outputs
                validProgress.set_description('VALIDATION Epoch [{}]'.format(
                    epoch+1
                ))
                validProgress.set_postfix(OrderedDict(
                loss=loss.item(),
                accuracy=correct/total*100
            ))

            validLoss /= len(validProgress)
            validLosses.append(validLoss)
            validAccs.append(correct/total)
        
        # lrScheduler.step()
        lrScheduler.step(metrics=validLoss)
        print('Validation loss: {}'.format(validLoss))
        
        if validLoss < bestValLoss:
            bestValLoss = validLoss
            torch.save({
                'epoch': epoch+1,
                'modelStateDict': model.state_dict(),
                'optimizerStateDict': optimizer.state_dict(),
                'validLoss': bestValLoss,
                'validAcc': bestValAcc
            }, model_dir)
            counter = 0
        else:
            counter += 1
            if counter >= 20:
                print('Validation loss has not improved for 20 epochs. Early stopping...')
                torch.save({
                    'epoch': epoch+1,
                    'modelStateDict': model.state_dict(),
                    'optimizerStateDict': optimizer.state_dict(),
                    'validLoss': bestValLoss,
                    'validAcc': bestValAcc
                }, 'lastModel.pt')
                print('Last model saved')
                break
        
        if correct/total*100 > bestValAcc:
            bestValAcc = correct/total*100
            torch.save({
                'epoch': epoch+1,
                'modelStateDict': model.state_dict(),
                'optimizerStateDict': optimizer.state_dict(),
                'validLoss': bestValLoss,
                'validAcc': bestValAcc
            }, 'bestAccModel.pt')
    
    print('Training Completed.')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(trainLosses) + 1), trainLosses, label='Training Loss')
    plt.plot(range(1, len(validLosses) + 1), validLosses, label='Validation Loss')
    plt.plot(range(1, len(validAccs) + 1), validAccs, label='Validation Accuracy')
    plt.plot(np.argmin(validLosses)+1, min(validLosses), 'ob')
    plt.plot(np.argmax(validAccs)+1, max(validAccs), 'ok')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss_Trend.png')
    plt.close()

    print('Computing accuracy on training data...')
    # find training accuracy
    # load best saved model
    model = classifier(numClasses=len(CLASS_LABELS), modelWeights=torch.load(model_dir)['modelStateDict'])
    model = model.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainDataLoader:
            images, labels = data[0], data[1]
            images = images.to(DEVICE)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted+1) == labels).sum().item()
            del data, outputs
    
    trainAcc = correct/total*100
    print('Accuracy on training data with best model: {:.2f}%'.format(trainAcc))
    
    # find test accuracy
    print('Computing accuracy on test data...')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testDataLoader:
            images, labels = data[0], data[1]
            images = images.to(DEVICE)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted+1) == labels).sum().item()
            del data, outputs
    
    testAcc = correct/total*100
    print('Accuracy on test data with best model: {:.2f}%'.format(testAcc))

    model = classifier(numClasses=len(CLASS_LABELS), modelWeights=torch.load('bestAccModel.pt')['modelStateDict'])
    model = model.to(DEVICE)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainDataLoader:
            images, labels = data[0], data[1]
            images = images.to(DEVICE)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted+1) == labels).sum().item()
            del data, outputs
    
    tAcc = correct/total*100
    print('Accuracy on training data with best accuracy model: {:.2f}%'.format(tAcc))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testDataLoader:
            images, labels = data[0], data[1]
            images = images.to(DEVICE)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted+1) == labels).sum().item()
            del data, outputs
    
    testAcc = correct/total*100
    print('Accuracy on test data with best accuracy model: {:.2f}%'.format(testAcc))

    return trainAcc


def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    pass



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pt', help='the pre-trained model')
    opt = parser.parse_args()

    buildClassLabelDictionaries()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)






