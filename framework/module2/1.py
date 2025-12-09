import numpy as np
from tensor import Tensor

print("=" * 60)
print("ТЕСТ 1: Проверка базовой работы автоградиента")
print("=" * 60)

x = Tensor([1.0, 2.0, 3.0], autograd=True)
y = Tensor([4.0, 5.0, 6.0], autograd=True)

print(f"Тензоры:")
print(f"  x = {x.data}")
print(f"  y = {y.data}")

z = x + y
print(f"\nz = x + y")
print(f"  z = {z.data}")
print(f"  Ожидаемый результат: [5. 7. 9.]")

print(f"\nСтруктура:")
print(f"  z.creators: {[creator.id for creator in z.creators]}")
print(f"  z.operation_on_creation: '{z.operation_on_creation}'")
print(f"  x.children: {x.children}")
print(f"  y.children: {y.children}")

print(f"\nbackward для z")
z.backward()

print(f"\nГрадиенты после backward():")
print(f"  z.grad: {z.grad.data if z.grad is not None else 'None'}")
print(f"  x.grad: {x.grad.data if x.grad is not None else 'None'}")
print(f"  y.grad: {y.grad.data if y.grad is not None else 'None'}")

expected_grad = np.array([1.0, 1.0, 1.0])
x_grad_correct = np.array_equal(x.grad.data, expected_grad) if x.grad is not None else False
y_grad_correct = np.array_equal(y.grad.data, expected_grad) if y.grad is not None else False

print(f"\nРезультаты проверки:")
print(f"  Градиент x корректен: {x_grad_correct} (ожидается: {expected_grad})")
print(f"  Градиент y корректен: {y_grad_correct} (ожидается: {expected_grad})")

