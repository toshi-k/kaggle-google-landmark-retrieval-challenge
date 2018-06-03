from torch.autograd import Variable
from lib.img_loader import ImgLoader


class ImgEmbedder:

    def __init__(self, model, dir_images):
        self.model = model
        self.loader = ImgLoader(dir_images)

    def get_vector(self, file_name):
        input_tensor = self.loader.load_image(file_name).cuda()
        output = self.model.forward(Variable(input_tensor))
        output_data = output.data.cpu().numpy()[0]

        return output_data
