import numpy as np

class Tensor(object):
    def __init__(self, data, creators=None, operation_on_creation=None):
        '''
        Конструктор
        
        :param self: ссылка на текущий объект
        :param data: данные для тензора
        :param creators: родительские тензоры, из которых был создан текущий
        :param operation_on_creation: операция, которая создала этот тензор
        :param grad: градиент тензора 
        '''
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
    
    def __add__(self, other):
        '''
        Перегрузка операции +
        
        :param self: ссылка на текущий объект
        :param other: другой объект тензора
        '''
        return Tensor(self.data + other.data, [self, other], "+")
    
    def __str__(self):
        '''
        Удобный вывод
            
        :param self: ссылка на текущий объект
        '''
        return str(self.data.__str__())
    
    def backward(self, grad):
        '''
        Реализация обратного распространения
        
        :param self: ссылка на текущий объект
        :param grad: градиент тензора 
        '''
        
        if grad is None:
            # что-то сделать
            pass    
    
        if self.creators is None:
            # что-то сделать
            pass
    
        self.grad = grad
        
        if self.operation_on_creation == "+":
            for i in range(len(self.creators)):
                self.creators[i].backward(grad)


def example_1():
    print("\n" + "="*50 + "\n")
    t_1 = Tensor([3, 15, 10])
    t_2 = Tensor([5, 6, 7])
    print(f"t_1 = {t_1}")
    print(f"t_2 = {t_2}")

    t_3 = t_1 + t_2
    print(f"t_3 = t_1 + t_2 = {t_3}")

    t_3.backward(Tensor([1, 2, 3]))
    print(f"Градиент t_1: {t_1.grad}")
    print(f"Градиент t_2: {t_2.grad}")
    print(f"Операция для t_3: {t_3.operation_on_creation}")


def example_2():
    print("\n" + "="*50 + "\n")
    a_1 = Tensor([1, 2, 3])
    a_2 = Tensor([1, 2, 3])
    a_3 = Tensor([1, 2, 3])
    a_4 = Tensor([1, 2, 3])

    print(f"a_1 = {a_1}")
    print(f"a_2 = {a_2}")
    print(f"a_3 = {a_3}")
    print(f"a_4 = {a_4}")

    a_add_1 = a_1 + a_2
    print(f"a_add_1 = a_1 + a_2 = {a_add_1}")

    a_add_2 = a_3 + a_4
    print(f"a_add_2 = a_3 + a_4 = {a_add_2}")

    a_add_3 = a_add_1 + a_add_2
    print(f"a_add_3 = a_add_1 + a_add_2 = {a_add_3}")

    a_add_3.backward(Tensor([4, 5, 3]))
    print(f"Градиент a_1: {a_1.grad}")
    print(f"Градиент a_2: {a_2.grad}")
    print(f"Градиент a_3: {a_3.grad}")
    print(f"Градиент a_4: {a_4.grad}")
    print(f"Градиент a_add_1: {a_add_1.grad}")
    print(f"Градиент a_add_2: {a_add_2.grad}")
    print(f"Градиент a_add_3: {a_add_3.grad}")



example_1()
example_2()
