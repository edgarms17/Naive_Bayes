![logo](/imagenes/tec.png)
>#**Tecnológico Nacional de México  
Instituto Tecnológico Campus Tijuana  
Ing. en informática/sistemas**  
**Asignatura**  
Datos Masivos  
**Semestre**  
Febrero- Junio 2022  
**Integrantes**  
Munguia Silva Edgar Geovanny  
Pérez López Alicia Guadalupe  
**Tema: Naive Bayes**  


>**¿Qué es ?**

 Es un clasificador probabilístico fundamentado en el teorema de Bayes y algunas hipótesis simplificadoras adicionales.
 - El modelo se denomina naïve porque trata todas las variables de predicción propuestas como independientes unas de otras. El bayesiano ingenuo es un algoritmo rápido y escalable que calcula las probabilidades condicionales para las combinaciones de atributos y el atributo de objetivo. 


>#**Pasos para llevar a cabo el algoritmo**

- El conjunto de datos en una tabla de frecuencias.
- Crear una tabla de probabilidad calculando las correspondientes a que ocurran los diversos eventos.
- La ecuación Naive Bayes se usa para calcular la probabilidad posterior de cada clase.
- La clase con la probabilidad posterior más alta es el resultado de la predicción.


##**Formula de algoritmo con el significado de sus variables**
![Formula](/imagenes/Formula.png)

##**Puntos Fuertes**
- Una manera fácil y rápida de predecir clases, para problemas de clasificación binarios y multiclase.
- El algoritmo se comporta mejor que otros modelos de clasificación, incluso con menos datos de entrenamiento.
- El desacoplamiento de las distribuciones de características condicionales de clase significan que cada distribución puede ser estimada independientemente como si tuviera una sola dimensión.

##**Puntos debiles**
- Los algoritmos Naive Bayes son conocidos por ser pobres estimadores. Por ello, no se deben tomar muy en serio las probabilidades que se obtienen.
- La presunción de independencia Naive muy probablemente no reflejará cómo son los datos en el mundo real.
- Cuando el conjunto de datos de prueba tiene una característica que no ha sido observada en el conjunto de entrenamiento, el modelo le asignará una probabilidad de cero y será inútil realizar predicciones.


