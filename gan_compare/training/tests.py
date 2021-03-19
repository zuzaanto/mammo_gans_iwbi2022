import torch
from torch import nn
embedding = nn.Embedding(4,3)
#embedding_result = embedding(torch.LongTensor([3,2]))
#rand_int = torch.randint(3,5,(2,))
#print (f"{rand_int}")
#embedding_result = embedding(rand_int)

rand_int = torch.randint(4,5,(100,))
print (f"{rand_int}")
embedding_result = embedding(rand_int)

print (f"{embedding_result}")