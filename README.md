![logo](/imagenes/tec.png)
># **Tecnológico Nacional de México**
Instituto Tecnológico Campus Tijuana  
Ing. en informática/sistemas  
**Asignatura**  
Datos Masivos  
**Semestre**  
Febrero- Junio 2022  
**Integrantes**  
Munguia Silva Edgar Geovanny  
Pérez López Alicia Guadalupe  
**Tema: Naive Bayes**  


> ##                     **¿Qué es Naive Bayes?**

 Es un clasificador probabilístico fundamentado en el teorema de Bayes y algunas hipótesis simplificadoras adicionales.
 - El modelo se denomina naïve porque trata todas las variables de predicción propuestas como independientes unas de otras. El bayesiano ingenuo es un algoritmo rápido y escalable que calcula las probabilidades condicionales para las combinaciones de atributos y el atributo de objetivo. 
 
 Para más informacion visitar el video en las referencias.


>##                     **Pasos para llevar a cabo el algoritmo**

- El conjunto de datos en una tabla de frecuencias.
- Crear una tabla de probabilidad calculando las correspondientes a que ocurran los diversos eventos.
- La ecuación Naive Bayes se usa para calcular la probabilidad posterior de cada clase.
- La clase con la probabilidad posterior más alta es el resultado de la predicción.


>##                  **Formula de algoritmo con el significado de sus variables**
![Formula](/imagenes/Formula.png)

>##                     **Puntos Fuertes**
- Una manera fácil y rápida de predecir clases, para problemas de clasificación binarios y multiclase.
- El algoritmo se comporta mejor que otros modelos de clasificación, incluso con menos datos de entrenamiento.
- El desacoplamiento de las distribuciones de características condicionales de clase significan que cada distribución puede ser estimada independientemente como si tuviera una sola dimensión.

>##                      **Puntos debiles**
- Los algoritmos Naive Bayes son conocidos por ser pobres estimadores. Por ello, no se deben tomar muy en serio las probabilidades que se obtienen.
- La presunción de independencia Naive muy probablemente no reflejará cómo son los datos en el mundo real.
- Cuando el conjunto de datos de prueba tiene una característica que no ha sido observada en el conjunto de entrenamiento, el modelo le asignará una probabilidad de cero y será inútil realizar predicciones.

##### Referencias bibliograficas 

- Roman, V. (2021, 9 diciembre). Algoritmos Naive Bayes: Fundamentos e Implementación. Medium. https://medium.com/datos-y-ciencia/algoritmos-naive-bayes-fudamentos-e-implementaci%C3%B3n-4bcb24b307f

- Gonzalez, L. (2020, 21 agosto). Naive Bayes – Teoría. 🤖 Aprende IA. Recuperado 11 de mayo de 2022, de https://aprendeia.com/naive-bayes-teoria-machine-learning/

- Cardellino, F. (2021, 28 abril). Cómo funcionan los clasificadores Naive Bayes: con ejemplos de código de Python. freeCodeCamp.org. Recuperado 11 de mayo de 2022, de https://www.freecodecamp.org/espanol/news/como-funcionan-los-clasificadores-naive-bayes-con-ejemplos-de-codigo-de-python/

- Gonzalez, A. C. L. (2019, 20 septiembre). NAIVE BAYES - TEORÍA | #46 Curso Machine Learning con Python [Vídeo]. YouTube. https://www.youtube.com/watch?v=949tYJgRvRg  

>##                       **Ejemplo en código**  
~~~
//Importar las librerias necesarias

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

//Cargar los datos especificando la ruta del archivo

val data = spark.read.format("libsvm").load("C:/spark/spark-2.4.8-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

println ("Numero de lineas en el archivo de datos:" + data.count ())

//Mostrar las primeras 20 líneas por defecto

data.show()

//Divida aleatoriamente el conjunto de datos en conjunto de entrenamiento y conjunto de prueba de acuerdo con los pesos proporcionados. También puede especificar una seed

val Array (trainingData, testData) = data.randomSplit (Array (0.7, 0.3), 100L)
// El resultado es el tipo de la matriz, y la matriz almacena los datos de tipo DataSet

//Incorporar al conjunto de entrenamiento (operación de ajuste) para entrenar un modelo bayesiano
val naiveBayesModel = new NaiveBayes().fit(trainingData)

//El modelo llama a transform() para hacer predicciones y generar un nuevo DataFrame.

val predictions = naiveBayesModel.transform(testData)

//Salida de datos de resultados de predicción
predictions.show()

//Evaluación de la precisión del modelo

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
// Precisión
val precision = evaluator.evaluate (predictions) 

//Imprimir la tasa de error
println ("tasa de error =" + (1-precision))
~~~
