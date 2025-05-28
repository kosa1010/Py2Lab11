# Wprowadzenie do SciPy w Pythonie
Przydatne linki
- https://docs.scipy.org/doc/scipy/
- https://numpy.org/
- https://scipy-lectures.org/

## 🔹 1. Wstęp – czym jest SciPy?

**SciPy** to biblioteka w Pythonie oparta na NumPy, oferująca szeroki zestaw narzędzi do obliczeń naukowych i technicznych:
- Algebra liniowa: `scipy.linalg`
- Optymalizacja: `scipy.optimize`
- Całkowanie numeryczne: `scipy.integrate`
- Interpolacja: `scipy.interpolate`
- Statystyka: `scipy.stats`
- Analiza sygnałów, FFT, geometria przestrzenna – i więcej

## 🔹 2. Instalacja
```bash
pip install scipy
```
## 🔹 3. Algebra liniowa z scipy.linalg
✅ Przykład 1a: Rozwiązywanie układu równań liniowych
`3x + 1y = 9`
`1x + 2y = 8`
```python
import numpy as np
from scipy.linalg import solve 

A = np.array([[3, 1], [1, 2]])     # Definicja macierzy współczynników A (lewa strona równań)
b = np.array([9, 8])               # Definicja wektora wyrazów wolnych b (prawa strona równań)
x = solve(A, b)                    # solve znajduje wektor x, który spełnia równanie Ax = b
print("Rozwiązanie:", x)
```
✅ Przykład 1b: Wyznacznik macierzy i macierz odwrotna
```python
from scipy.linalg import det, inv

A = np.array([[1, 2], [3, 4]])    # definicja macierzy 2x2 
print("Wyznacznik:", det(A))
print("Macierz odwrotna:\n", inv(A))
```
## 🔹 4. Optymalizacja z scipy.optimize
✅ Przykład 2: Znajdowanie minimum funkcji
![image](https://github.com/user-attachments/assets/59815d66-2261-4469-a2c2-ea4e2538491f)
```python
from scipy.optimize import minimize

def f(x):
    return (x - 2)**2 + 1

result = minimize(f, x0=0)          # f — funkcja, którą minimalizujemy. x0=0 — punkt startowy algorytmu optymalizacji (czyli zgadujemy, że minimum może być gdzieś w pobliżu 0)
print("Minimum przy x =", result.x)
```
✅ Przykład 2b: Minimum funkcji dwóch zmiennych
![image](https://github.com/user-attachments/assets/22890d1d-18b9-4c7f-947b-3a8be9cbe5d2)

```python
def f(v):
    x, y = v
    return x**2 + y**2 + x*y

result = minimize(f, x0=[1, 1])      # w tym przypadku x0=[1, 1] to punkt startowy, czyli algorytm zacznie szukać minimum w okolicach punktu (1,1)
print("Minimum:", result.x)
```
✅ Przykład 2c: Znajdowanie miejsca zerowego dla równania 
![image](https://github.com/user-attachments/assets/ed15e18d-017a-4c45-85f0-1a3de646b804)
```python
from scipy.optimize import root
import numpy as np

def f(x):
    return np.cos(x) - x

sol = root(f, x0=0.5)            # x0=0.5 – punkt startowy, od którego algorytm zaczyna szukać rozwiązania
print("Pierwiastek:", sol.x[0])
```
## 🔹 5. Całkowanie numeryczne z scipy.integrate
Poniższy kod numerycznie oblicza oznaczoną całkę funkcji ![image](https://github.com/user-attachments/assets/da3c0e68-85fc-4bd1-926f-5d3e68a584e7)  w przedziale od 0 do 2.
✅ Przykład 3: Całka oznaczona
```python
from scipy.integrate import quad

def f(x):
    return x**2

result, _ = quad(f, 0, 2)
print("Całka:", result)
```
Oblicza wartość całki oznaczonej:
![image](https://github.com/user-attachments/assets/e8d8be0b-4edc-42d2-b3d9-dd6d83d4d053)
`quad` zwraca dwie wartości:  
-  `resul`t: wartość całki,  
-  `_`: oszacowanie błędu (tu pomijane przez `_`).

## 🔹 6. Interpolacja z scipy.interpolate
✅ Przykład 4: Interpolacja liniowa – czyli oszacowanie wartości funkcji pomiędzy znanymi punktami danych.
```python
from scipy.interpolate import interp1d
import numpy as np

# oznaczenie znanych punktów funkcji f(0)=0, f(1)=1, f(2)=0, f(3)=1
x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 0, 1])

f = interp1d(x, y)
print("Interpolowana wartość dla 1.5:", f(1.5))
```
✅ Przykład 4b: Interpolacja kubiczna (sześcienna)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.linspace(0, 10, 5)                # 5 punktów równomiernie rozłożonych od 0 do 10
y = np.sin(x)                            # wartości funkcji sinus dla tych punktów

f_cubic = interp1d(x, y, kind='cubic')   # tworzy funkcję interpolującą metodą sześciennych wielomianów (kubiczną)

x_new = np.linspace(0, 10, 100)          # tworzy gęstą siatkę x_new – 100 punktów od 0 do 10
y_new = f_cubic(x_new)                   # oblicza interpolowane wartości y_new dla tych punktów

plt.plot(x, y, 'o', label='punkty')
plt.plot(x_new, y_new, '-', label='interpolacja kubiczna')
plt.legend()
plt.show()
```
## 🔹 7. Statystyka z scipy.stats
✅ Przykład 5: Statystyki opisowe
```python
from scipy import stats
import numpy as np

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])
print("Średnia:", np.mean(data))
print("Mediana:", np.median(data))
print("Odchylenie:", np.std(data))
print("Moda/Dominanta:", stats.mode(data, keepdims=True).mode[0])   # najczęściej występującą wartość
```
✅ Przykład 5b: Rozkład normalny – wykres
```python
import matplotlib.pyplot as plt
from scipy.stats import norm

mu, sigma = 0, 1                    # definiuje średnią i odchylenie standardowe rozkładu normalnego
x = np.linspace(-4, 4, 100)         # tworzy 100 punktów od -4 do 4 (osi X)
pdf = norm.pdf(x, mu, sigma)        # oblicza wartości funkcji gęstości prawdopodobieństwa rozkładu normalnego w tych punktach

plt.plot(x, pdf)
plt.title("Rozkład normalny (μ=0, σ=1)")
plt.grid(True)
plt.show()
```
## Zadania do  wykonania
1. Rozwiąż układ równań
   
 ![image](https://github.com/user-attachments/assets/13b2a1ed-0011-48ab-ac4c-e46740be2520)
 
Znajdź `x` i `y` używając `scipy.linalg.solve`
2. Znajdź minimum funkcji ![image](https://github.com/user-attachments/assets/106eca2f-fc8e-4878-bc6f-2feb9737f6af)
3. Oblicz całkę oznaczoną z funkcji `f(x) = sin(x)` od `0` do `π`
4. Interpolacja liniowa
x = [0, 2, 4], y = [1, 3, 2], znajdź wartość dla x = 3
5. Statystyka dla [1, 2, 2, 3, 4] (średnia, odchylenie standardowe, mediana, wartosć najcześciej wystepująca).
6. Odwrotność i wyznacznik macierzy
A = [[4, 7], [2, 6]]
7. Znajdź pierwiastek ![image](https://github.com/user-attachments/assets/47adff63-4f97-4a20-b3ee-c0c80de9542c)
