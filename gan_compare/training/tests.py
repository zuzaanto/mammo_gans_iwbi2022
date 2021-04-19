import torch
from torch import nn
import unittest


class UnitTests(unittest.TestCase):

    def test_torch_nn_Embedding(self):
        print("===================================================")
        print('Start test_torch_nn_Embedding unit test..')
        # 4 = number of embeddings, 3 = number of dimensions (i.e. vectors per embedding)
        embedding = nn.Embedding(4, 3)

        # Let's test nn.Embedding using torch LongTensor input
        long_tensor = torch.LongTensor([3, 2])
        embedding_result = embedding(long_tensor)
        print(f"{embedding_result}")
        self.assertTrue(embedding_result is not None)

        # Let's test nn.Embedding using torch random integer input
        rand_int = torch.randint(1, 4, (2,))
        embedding_result = embedding(rand_int)
        print(f"{embedding_result}")
        self.assertTrue(embedding_result is not None)

        # Let's test the allowed inpout dimensions of nn.Embedding using torch random integer input.
        # Generate a tensor of 100 rand_ints between 3, 5. Is this valid considering nn.Embedding(4, 3)?
        rand_int = torch.randint(3, 5, (100,))
        print(f"{rand_int}")

        try:
            embedding_result = embedding(rand_int)
            self.assertTrue(embedding_result is not None)
            print(f"{embedding_result}")

        except IndexError:
            error_message = 'This did not work (IndexError: index out of range in self) because torch nn.Embedding ' \
                            'start to count the num_embeddings at index 0 --> [0,max+1] --> nn.Embedding(max+1,dim) '
            print(f"{error_message}")

    def test_torch_gpu_availability(self):
        print("===================================================")
        print('Start test_torch_gpu_availability unit test..')
        import torch.cuda as tcuda
        is_gpu_available: bool = tcuda.is_available()
        if is_gpu_available:
            num_gpus = torch.cuda.device_count()
            currently_used_gpu = torch.cuda.current_device()
            names_of_gpus = ''
            for idx in range(num_gpus):
                names_of_gpus += torch.cuda.get_device_name(idx) + ' '
            print(f'is_gpu_available (?): {is_gpu_available}, currently_used_gpu: {currently_used_gpu}, '
                  f'names_of_gpus: {names_of_gpus}')
        self.assertTrue(is_gpu_available is not None)


    def test_output_shape_Con2d_layers(self):
        # You can use this formula [(W−K+2P)/S]+1. <-- https://stackoverflow.com/a/53580139
        # W is the input volume - in our case e.g. 128
        # K is the Kernel size  - in our case e.g. 6
        # P is the padding - in our case e.g. 1
        # S is the stride - in our case e.g. 2
        print("===================================================")
        print("discriminator:")
        print("Conv2d Layers")
        S = 2
        P = 2 #1
        K = 6 #4
        W = 128
        resulting_shape1 = ((W-K+2*P)/S)+1
        print("resulting_shape1", str(resulting_shape1))

        S = 2
        P = 2 #1
        K = 6 #4
        W = resulting_shape1
        resulting_shape2 = ((W-K+2*P)/S)+1
        print("resulting_shape2", str(resulting_shape2))

        S = 2
        P = 2 #1
        K = 6 #4
        W = resulting_shape2
        resulting_shape3 = ((W-K+2*P)/S)+1
        print("resulting_shape3", str(resulting_shape3))

        S = 2
        P = 2 #1
        K = 6 #4
        W = resulting_shape3
        resulting_shape4 = ((W-K+2*P)/S)+1
        print("resulting_shape4", str(resulting_shape4))

        S = 2
        P = 2 #1
        K = 6 #4
        W = resulting_shape4
        resulting_shape5 = ((W-K+2*P)/S)+1
        print("resulting_shape5", str(resulting_shape5))

        S = 1
        P = 1 #1
        K = 6 #4
        W = resulting_shape5
        resulting_shape6 = ((W-K+2*P)/S)+1
        print("resulting_shape6", str(resulting_shape6))

        # shape should be a natural number.
        self.assertTrue(resulting_shape1 %2==0 and resulting_shape2%2==0 and resulting_shape3%2==0 and resulting_shape4%2==0 and resulting_shape5%2==0)

    def test_output_shape_ConvTranspose2d_layers(self):
        print("===================================================")
        print("generator:")
        print("ConvTranspose2d Layers")
        # Below formula found here: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
        # and here: https://stackoverflow.com/a/58776482
        # H_out = (H_in−1)*stride[0] − 2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
        # W_out = (W_in−1)×stride[1] − 2×padding[1] + dilation[1]×(kernel_size[1]−1) + output_padding[1] + 1
        S = 1
        P = 0
        K = 4
        H_in = 1
        H_out = (H_in-1) * S - (2*P) + 1* (K-1) + 1
        W_in = 1
        W_out = (W_in-1) * S - (2*P) + 1* (K-1) + 1
        resulting_shape1 = "H=" + str(H_out) + ", W=" + str(W_out)

        print("resulting_shape1", str(resulting_shape1))

        S = 2
        P = 1
        K = 4
        H_in = H_out
        H_out = (H_in-1) *S - (2*P) + 1* (K-1) + 1
        W_in = W_out
        W_out = (W_in-1) *S - (2*P) + 1* (K-1) + 1
        resulting_shape2 = "H=" + str(H_out) + ", W=" + str(W_out)

        print("resulting_shape2", str(resulting_shape2))

        S = 2
        P = 1
        K = 3
        H_in = H_out
        H_out = (H_in-1) *S - (2*P) + 1* (K-1) + 1
        W_in = W_out
        W_out = (W_in-1) *S - (2*P) + 1* (K-1) + 1
        resulting_shape3 = "H=" + str(H_out) + ", W=" + str(W_out)

        print("resulting_shape3", str(resulting_shape3))

        S = 2
        P = 1
        K = 2
        H_in = H_out
        H_out = (H_in-1) *S - (2*P) + 1* (K-1) + 1
        W_in = W_out
        W_out = (W_in-1) *S - (2*P) + 1* (K-1) + 1
        resulting_shape4 = "H=" + str(H_out) + ", W=" + str(W_out)
        print("resulting_shape4", str(resulting_shape4))

        # shape should be a natural number.
        self.assertTrue(resulting_shape1.find(".")==-1 and resulting_shape2.find(".")==-1 and resulting_shape3.find(".")==-1 and resulting_shape4.find(".")==-1)


if __name__ == '__main__':
    unittest.main()