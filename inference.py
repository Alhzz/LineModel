from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import torch
import io
import json
from flask import Flask, jsonify, request, json

""" Prepre necesary thing """
# FLASK_DEBUG=0 FLASK_APP=inference.py flask run
app = Flask(__name__)

# Name of an exist class

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        print(file)
        # class_names = ['Crateva', 'Ficus', 'Tacca', 'Tiliacora', 'Uvaria']
        class_names = ['Alstonia', 'Alyxia', 'Andrographis', 'Antidesma', 'Averrhoa', 'Capparis', 'Cardiospermum', 'Cinchona', 'CordylineA', 'CordylineB', 'Crateva', 'Eurycoma', 'Ficus', 'Garcinia', 'Graptophyllum', 'Hrrisonia', 'Tacca', 'Tiliacora', 'Tinospora', 'Uvaria']

        # Check if a GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Transform input image into a proper model's input shape
        loader = transforms.Compose([transforms.Resize(256), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

        def image_loader(image_name):
            # load and manage an image
            image = Image.open(io.BytesIO(image_name))
            image = image.convert('RGB')
            image = loader(image)
            image = image.float()
            image = Variable(image, requires_grad=True)
            return image.to(device)



        """ Detection section """
        # Load model
        # model_path = "./model/resnet.pth"
        model_path = "./model/efficientnet2.pth"
        model = torch.load(model_path, map_location=device)

        # Load image
        image = image_loader(img_bytes)

        model.eval()

        outputs = model(image.unsqueeze(0))
        _, preds = torch.max(outputs, 1)
        print(class_names[preds])
        return jsonify({'class_name': class_names[preds]})

if __name__ == '__main__':
   app.run(debug=False, )