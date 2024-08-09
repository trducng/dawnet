# Understand the affects of hyperparameters on channel visualization

- Clamp the min and max of an image so that (1) any small sets of pixels cannot dominate the other pixels which results in mostly black image with a few pixels of white; (2) overcoming the quirks of NN and resulting a more interpretable images. **Though: it's interesting to understand the behaviors of NN if we don't clamp the optimized image**.
- With momentum (e.g. 0.9, 0.99...), the optimized images tend to have less checker-board patterns (maybe because momentum helps reduce the amount of high frequency signals).
- With weight decay, the image seems to be sharper. Yet, with higher weight decay, the image tends to become darker.

## Scratch

---

- Momemtum: 0.9
- Lr: 0.01

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
```

Result: `image001.png`

---

- Lr: 0.01
- Weight decay: 1e-5

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, weight_decay=1e-5)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
```

Result `image002.png`


---

- LR: 0.01
- Momentum: 0.9
- Weight decay: 1e-5

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-5)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
```

Result `image003.png`

---

- LR: 0.01
- Momentum: 0.9
- Weight decay: 1e-1

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-1)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
```

Result `image004.png`


---

- LR: 0.01
- Momentum: 0.9
- Weight decay: 1e-5
- Blur the image with kernel 3x3

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

before_blur = None
import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-5)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)
        before_blur = opt_image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
        opt_image = cv2.blur(before_blur, (3, 3), cv2.BORDER_DEFAULT)
        opt_image = torch.tensor(opt_image).permute(2, 0 ,1).unsqueeze(0).cuda()
        opt_image.requires_grad_()

    state_dict = optimizer.state_dict()
    optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-5)
    optimizer.load_state_dict(state_dict)
    optimizer.zero_grad()
        

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
plt.imshow((before_blur * 255).astype(np.uint8))
```

Result `image005.png`

---

- Optimize before RELU in the last layer
- Blurring kernel 3x3

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

before_blur = None
import torch.optim as optim
optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-5)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)
        before_blur = opt_image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
        opt_image = cv2.blur(before_blur, (3, 3), cv2.BORDER_DEFAULT)
        opt_image = torch.tensor(opt_image).permute(2, 0 ,1).unsqueeze(0).cuda()
        opt_image.requires_grad_()

    state_dict = optimizer.state_dict()
    optimizer = optim.SGD([opt_image], lr=0.01, momentum=0.9, weight_decay=1e-5)
    optimizer.load_state_dict(state_dict)
    optimizer.zero_grad()
        

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(temp)
```

Result `image006.png`

---
- Blurring kernel 5x5

Result `image007.png`

- Blurring kernel 7x7

Result `image008.png`

- Blurring kernel 5x5
- Weight decay: 1e-3

Result `image009.png`

- Blurring kernel 5x5
- Weight decay: 1e-1

Result `image010.png`

- Blurring kernel 3x3
- Weight decay: 1e-1

Result `image011.png`

---

- Use Adam
- LR: 0.01
- Weight decay: 1e-2

```python
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
opt_image = norm(torch.rand(1, 3, 299, 299)).cuda()
opt_image.requires_grad_()

before_blur = None
import torch.optim as optim
optimizer = optim.Adam([opt_image], lr=0.01, weight_decay=1e-2)
optimizer.zero_grad()

for idx in range(10000):
    out = model(opt_image)
    if idx % 500 == 0:
        print(idx, out.item())
    out.backward()
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        torch.clamp_(opt_image, min=0.0, max=1.0)
        before_blur = opt_image.cpu().detach().squeeze().permute(1, 2, 0).numpy()
        opt_image = cv2.blur(before_blur, (3, 3), cv2.BORDER_DEFAULT)
        opt_image = torch.tensor(opt_image).permute(2, 0 ,1).unsqueeze(0).cuda()
        opt_image.requires_grad_()

    state_dict = optimizer.state_dict()
    optimizer = optim.Adam([opt_image], lr=0.01, weight_decay=1e-5)
    optimizer.load_state_dict(state_dict)
    optimizer.zero_grad()
        

print(opt_image.min(), opt_image.max())
cpu_image = opt_image.cpu().detach().numpy()
cpu_image = (cpu_image - cpu_image.min()) / (cpu_image.max() - cpu_image.min())

temp = (cpu_image.squeeze().transpose(1, 2, 0) * 255).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow((before_blur * 255).astype(np.uint8))
```

Result `image012.png`

