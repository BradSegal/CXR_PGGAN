import torchvision.datasets as datasets  # Dataset Wrappers for Standardizing Data
import torchvision.transforms as transforms  # Image Transformation for Resizing, etc


class VariableImageFolder(datasets.ImageFolder):
    def __init__(self, dataset, img_size=4, mean=0, std=1, channels=3):
        super(VariableImageFolder, self).__init__(dataset, transform=transforms)

        self.mean = [mean] * channels
        self.std = [std] * channels
        self.img_size = int(img_size)
        self.max = super(VariableImageFolder, self).__len__()
        self.imgs = self.max
        self.update_transformations()  # Set the default method for transforming images

    def __len__(self):
        if self.imgs > self.max:
            return self.max
        else:
            return self.imgs

    def update_img_len(self, new_len):
        new_len = int(new_len)
        if new_len > 0:
            self.imgs = new_len
        else:
            self.imgs = 1  # Corrects for if during loading the image requirements are negative

    def update_img_size(self, new_size):
        self.img_size = int(new_size)
        self.update_transformations()

    def update_transformations(self):
        transformations = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(tuple(self.mean), tuple(self.std))])
        self.transform = transformations
