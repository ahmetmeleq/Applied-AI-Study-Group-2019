import torch
import torchvision

# An instance of your model.
model = torchvision.models.resnet18(pretrained=True)

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, inp):
        if inp.sum() > 0:
            output = self.weight.mv(inp)
        else:
            output = self.weight + inp
        return output

my_module = MyModule(20, 30)
sm = torch.jit.script(my_module)

traced_script_module.save("traced.pt")
sm.save("sm.pt")
