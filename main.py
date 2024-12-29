from SearchSpace import *
from SearchStrategy import *
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=96, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = DataLoader(testset, batch_size=96, shuffle=False, num_workers=2)

OPERATIONS = [
    '3x3_sepconv',
    '5x5_sepconv',
    '3x3_conv',
    '3x3_dilconv',
    '3x3_avgpool',
    '3x3_maxpool',
    '5x5_maxpool',
    '7x7_maxpool',
    '1x3_3x1_conv',
    '1x7_7x1_conv',
    '1x1_conv',
    '3x3_conv',
    'identity'
]
combine_methods = nn.ModuleList([
    Add(),
    Concat()
])
# SearchSpace 설정
search_space = SearchSpace(
    input_size=3,
    output_size=10,
    BackBone=[1, 1,],
    ops=OPERATIONS,
    combine_methods=combine_methods,
    B=5,
    init_channels=16
)

search_space_sizes = range(20, 80, 20)


ga_results = []
rs_results = []
ppo_results = []
ga_std_dev = []
rs_std_dev = []
ppo_std_dev = []
trainandtest = TrainAndTest(trainloader, testloader)
controller = Controller(hidden_size = 64, num_blocks = 5, num_operations = 13, num_combine_methods = 2)
trainer = SimplePPOTrainer(search_space, controller, SynFlow(trainloader, testloader), False)
evolution_search = Regularized_Evolution(search_space, 25, SynFlow(trainloader, testloader))
random_search = Random_Search(search_space, SynFlow(trainloader, testloader))

for size in search_space_sizes:

    ga_losses = []
    rs_losses = []
    ppo_losses = []
    num_repeats = 10

    for _ in range(num_repeats):

        es_best = evolution_search.search(size, int(size * 0.75))
        ga_loss = trainandtest.estimate(es_best)
        ga_losses.append(ga_loss)


        rs_best = random_search.search(size)
        rs_loss = trainandtest.estimate(rs_best)
        rs_losses.append(rs_loss)

        ppo_best = trainer.sample(size)
        ppo_loss = trainandtest.estimate(ppo_best)
        ppo_losses.append(ppo_loss)

    ga_results.append(np.mean(ga_losses))
    ga_std_dev.append(np.std(ga_losses))
    rs_results.append(np.mean(rs_losses))
    rs_std_dev.append(np.std(rs_losses))
    ppo_results.append(np.mean(ppo_losses))
    ppo_std_dev.append(np.std(ppo_losses))

plt.figure(figsize=(10, 6))
plt.errorbar(search_space_sizes, ga_results, yerr=ga_std_dev, fmt='-o', capsize=5, label="Genetic Algorithm")
plt.errorbar(search_space_sizes, rs_results, yerr=rs_std_dev, fmt='-o', capsize=5, label="Random Search")
plt.errorbar(search_space_sizes, ppo_results, yerr=ppo_std_dev, fmt='-o', capsize=5, label="RNN Controller with PPO")
plt.xlabel("Search Space Size")
plt.ylabel("Loss")
plt.title("Performance Comparison with Error Bars")
plt.legend()
plt.grid(True)
plt.show()