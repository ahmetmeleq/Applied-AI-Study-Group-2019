#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

	if (argc != 2) {
		std::cerr << "app <module>" << std::endl;
		return -1;
	}

	torch::jit::script::Module module;
	try {
		module = torch::jit::load(argv[1]);
	} catch (const c10::Error &e) {
		std::cerr << "Error Loading the Model" << std::endl;
		return -1;
	}

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({1, 3, 224, 224}));

	at::Tensor output = module.forward(inputs).toTensor();
	std::cout << output << std::endl;
	return 0;
  //torch::Tensor tensor = torch::rand({2, 3});
  //std::cout << tensor << std::endl;
}
