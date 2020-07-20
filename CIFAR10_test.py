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
		return predictions


def main():
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=4, shuffle=True)

	test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=4, shuffle=False)

	classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	net = Net(num_of_cells = 3)

	criterion = nn.CrossEntropyLoss(reduction='none')
	optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

	# Training the Network
	for epoch in range(1, 9):
		for iteration, data in enumerate(train_loader):
			inputs, labels = data
			optimizer.zero_grad()
			hypotheses = net.forward_multiple_hypotheses(inputs)
			for idx in range(len(hypotheses)):
				outputs = hypotheses[idx]
				_, predictions = torch.max(outputs.data, 1)
				net.VoronoiCells[idx].fill_class_numbers(predictions, labels)
				loss = criterion(outputs, labels)
				true_classified = (predictions == labels).int()
				false_classified = (~(predictions == labels)).int()
				epsilon = 0.2
				loss = torch.add(torch.mul(loss, true_classified) * (1 - epsilon), 
					torch.mul(loss, false_classified) * (epsilon / (net.number_of_Voronoi_Cells - 1))).sum()
				loss.backward(retain_graph=True)
				optimizer.step()

				if iteration % 2000 == 1999:
					net.VoronoiCells[idx].cell_analysis(classes)
		print("Epoch % has finished." % (epoch))
	print("Finished Training")
if __name__ == "__main__":
	main()

