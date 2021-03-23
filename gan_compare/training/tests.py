import torch
from torch import nn
import unittest


class UnitTests(unittest.TestCase):

    def test_torch_nn_Embedding(self):
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

    def test_torch_GPU_availability(self):
        print('Start test_torch_GPU_availability unit test..')
        import torch.cuda as tcuda
        is_GPU_available: bool = tcuda.is_available()
        if is_GPU_available:
            num_GPUs = torch.cuda.device_count()
            currently_used_GPU = torch.cuda.current_device()
            for idx in range(num_GPUs):
                names_of_GPUs += torch.cuda.get_device_name(idx) + ' '
            print(f'is_GPU_available (?): {is_GPU_available}, currently_used_GPU: {currently_used_GPU}, names_of_GPUs: {names_of_GPUs}')
        self.assertTrue(is_GPU_available is not None)


if __name__ == '__main__':
    unittest.main()
