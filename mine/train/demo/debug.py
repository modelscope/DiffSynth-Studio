import inspect

def g(data):
    return data * data

def z(x):
    def f(x):
        lst = []
        for i in range(x):
            val = g(i)
            lst.append(val)
        return lst
    return f(x)
z(3)
   