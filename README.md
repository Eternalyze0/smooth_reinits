# Smooth Re-Initializations

The last layer is injected with noise that decays over time. Good performance on cartpole is achieved despite epsilon being set to zero.

Alternatively the middle layer is injected with (large!) noise that never decays and a skip connection is added.

## nanoGPT baseline

step 2000: train loss 1.7648, val loss 1.8857

## nanoGPT smooth reinits

## dqn_last_layer_decay.py

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/2648a262-eb37-485f-8995-77505a60d0b5" />

## dqn_middle_layer_no_decay.py

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/e68acc0c-1fff-49ac-9a0f-a76634846adc" />


## Code

```py
    def forward(self, x):
        if self.experimental:
            with torch.no_grad():
                # self.fc1.weight.data.add_(torch.randn_like(self.fc1.weight) * self.fc1.weight * self.scaling_factor)
                # self.fc2.weight.data.add_(torch.randn_like(self.fc2.weight) * self.fc2.weight * self.scaling_factor)
                self.fc3.weight.data.add_(torch.randn_like(self.fc3.weight) * self.fc3.weight * self.scaling_factor)
                # self.fc1.bias.data.add_(torch.randn_like(self.fc1.bias) * self.fc1.bias * self.scaling_factor)
                # self.fc2.bias.data.add_(torch.randn_like(self.fc2.bias) * self.fc2.bias * self.scaling_factor)
                self.fc3.bias.data.add_(torch.randn_like(self.fc3.bias) * self.fc3.bias * self.scaling_factor)
                self.scaling_factor *= 0.5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```py
    def forward(self, x):
        if self.experimental:
            with torch.no_grad():
                # self.fc1.weight.data.add_(torch.randn_like(self.fc1.weight) * self.fc1.weight * self.scaling_factor)
                self.fc2.weight.data.add_(torch.randn_like(self.fc2.weight) * self.fc2.weight * self.scaling_factor)
                # self.fc3.weight.data.add_(torch.randn_like(self.fc3.weight) * self.fc3.weight * self.scaling_factor)
                # self.fc1.bias.data.add_(torch.randn_like(self.fc1.bias) * self.fc1.bias * self.scaling_factor)
                self.fc2.bias.data.add_(torch.randn_like(self.fc2.bias) * self.fc2.bias * self.scaling_factor)
                # self.fc3.bias.data.add_(torch.randn_like(self.fc3.bias) * self.fc3.bias * self.scaling_factor)
                self.scaling_factor *= 1.0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x
```

## References

Inspiration from https://arxiv.org/abs/2205.07802.

Baseline from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py.
