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

package org.apache.spark.examples.mllib    

import org.apache.spark.{SparkConf, SparkContext}    
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}    
import org.apache.spark.mllib.util.MLUtils    


object NaiveBayesExample {    

  def main(args: Array[String]): Unit = {    
    val conf = new SparkConf().setAppName("NaiveBayesExample")    
    val sc = new SparkContext(conf)   
    
    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

    // Split data into training (60%) and test (40%).
    val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(training, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    // Save and load model
    model.save(sc, "target/tmp/myNaiveBayesModel")
    val sameModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")
    
    sc.stop()
  }
}  
