from __future__ import print_function
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VoronoiCell():
	def __init__(self, cell_id):
		self.id = cell_id
		# Linear Layers
		self.Linear_1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
		self.Linear_2 = nn.Linear(in_features = 120, out_features = 84)
		self.Linear_3 = nn.Linear(in_features = 84, out_features = 10)

		self.correct_class_numbers = torch.zeros(size = (10, ))
		self.total_class_numbers = torch.zeros(size = (10, ))

	def forward(self, inp):
		return self.Linear_3(F.relu(self.Linear_2(F.relu(self.Linear_1(inp)))))	

	def fill_class_numbers(self, prediction_tensor, label_tensor):
		for idx in range(prediction_tensor.size()[0]):
			self.total_class_numbers[label_tensor[idx]] += 1
			if prediction_tensor[idx] == label_tensor[idx]:
				self.correct_class_numbers[prediction_tensor[idx]] += 1

	def cell_analysis(self, classes):
		average_accuracy = 0.0
		print("Head {}:".format(self.id))
		for idx in range(self.correct_class_numbers.size()[0]):
			average_accuracy += (100 * self.correct_class_numbers[idx] / self.total_class_numbers[idx]) / self.correct_class_numbers.size()[0]
			print("\t The classification accuracy on " + classes[idx] + " images is %5.2f %%." %(100 * self.correct_class_numbers[idx] / self.total_class_numbers[idx]))
		print("\t The average accuracy is: %5.2f %%." % (average_accuracy))

		self.correct_class_numbers = torch.zeros(size = (10, ))
		self.total_class_numbers = torch.zeros(size = (10, ))

class Net(nn.Module):
	def __init__(self, num_of_cells):
		super(Net, self).__init__()

		# Convolutional Layers
		self.Conv2d_1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = (5, 5))
		self.Conv2d_2 = nn.Conv2d(in_channels =6, out_channels = 16, kernel_size = (5, 5))

		# Linear Layers
		self.Linear_1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
		self.Linear_2 = nn.Linear(in_features = 120, out_features = 84)
		self.Linear_3 = nn.Linear(in_features = 84, out_features = 10)

		# Pooling Layer
		self.MaxPool2d = nn.MaxPool2d(kernel_size = (2, 2))

		# Voronoi Cells
		self.number_of_Voronoi_Cells = num_of_cells
		self.VoronoiCells = []
		for cell_id in range(self.number_of_Voronoi_Cells):
			self.VoronoiCells.append(VoronoiCell(cell_id))

	def forward(self, inp):
		inp = self.MaxPool2d(F.relu(self.Conv2d_1(inp)))
		inp = self.MaxPool2d(F.relu(self.Conv2d_2(inp)))
		inp = inp.view(-1, 16 * 5 * 5)
		inp = F.relu(self.Linear_2(F.relu(self.Linear_1(inp))))
		return self.Linear_3(inp)

	def forward_multiple_hypotheses(self, inp):
		inp = self.MaxPool2d(F.relu(self.Conv2d_1(inp)))
		inp = self.MaxPool2d(F.relu(self.Conv2d_2(inp)))
		inp = inp.view(-1, 16 * 5 * 5)
		predictions = []
		for i in range(self.number_of_Voronoi_Cells):
			predictions.append(self.VoronoiCells[i].forward(inp))
		predictions = torch.stack(tensors=(tuple(predictions)), dim=0)
		return predictions

	def compute_head_losses(self, hypotheses, labels, criterion):
		loss_values = []
		for idx in range(hypotheses.size()[0]):
			loss = criterion(hypotheses[idx], labels)
			loss_values.append(loss)
		loss_values = torch.stack(tensors=(tuple(loss_values)), dim=0)
		return loss_values

	def compute_meta_losses(self, hypotheses, labels, criterion, min_indices, epsilon):
		for head_idx in range(self.number_of_Voronoi_Cells):
			delta_values = torch.ones(size=(hypotheses.size()[1], ))	
			for point_idx in range(hypotheses.size()[1]):
				if min_indices[point_idx] == head_idx:
					delta_values[point_idx] *= 1 - epsilon
				else:
					delta_values[point_idx] *= epsilon / (self.number_of_Voronoi_Cells - 1)

			loss = criterion(hypotheses[head_idx], labels)
			loss = torch.mul(loss, delta_values).sum()
			loss.backward(retain_graph=True)



def main():
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True)

	test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False)

	classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = Net(num_of_cells = 10)

	# criterion = nn.CrossEntropyLoss()
	criterion = nn.CrossEntropyLoss(reduction='none')
	optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

	# Training the Network
	for epoch in range(1, 9):
		for iteration, data in enumerate(train_loader):
			inputs, labels = data
			optimizer.zero_grad()
			hypotheses = net.forward_multiple_hypotheses(inputs) # size = (num_of_heads, batch_size, num_of_classes)
			head_losses = net.compute_head_losses(hypotheses, labels, criterion)
			min_values, min_indices = torch.min(input=head_losses, dim=0)
			epsilon = 0.2
			net.compute_meta_losses(hypotheses, labels, criterion, min_indices, epsilon)
			optimizer.step()
		print("Epoch % has finished." % (epoch))
	print("Finished Training")

	# Testing the Network
	with torch.no_grad():
		for data in test_loader:
			inputs, labels = data
			hypotheses = net.forward_multiple_hypotheses(inputs)
			for idx in range(hypotheses.size()[0]):
				outputs = hypotheses[idx]
				_, predictions = torch.max(outputs.data,  1)
				net.VoronoiCells[idx].fill_class_numbers(predictions, labels)

		for idx in range(hypotheses.size()[0]):
			net.VoronoiCells[idx].cell_analysis(classes)
	print("Finished Testing")

if __name__ == "__main__":
	main()