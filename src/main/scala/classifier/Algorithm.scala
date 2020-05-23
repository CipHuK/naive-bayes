package classifier

import scala.math.log

trait FeatureProbability[T, K] {
  // T - feature
  // K - category
}

private[classifier] class Model[K](lengths: Map[K, Int],
                                   docCount: Map[K, Int],
                                   wordCount: Map[K, Map[String, Int]],
                                   dictionarySize: Int) {

  def wordProbability(category: K, word: String): Double =
    log((wordCount(category).getOrElse(word, 0) + 1.0) / (lengths(category).toDouble + dictionarySize))

  def classProbability(category: K): Double = log(docCount(category).toDouble / docCount.values.sum)

  def classes: Set[K] = docCount.keySet
}

private[classifier] class Classifier[K](model: Model[K]) {
  def classify(str: String): K =
    model.classes.toList.map(c => (c, calculateProbability(c, str))).maxBy(_._2)._1

  def tokenize(str: String): Array[String] = str.split(' ')

  def calculateProbability(category: K, documentText: String): Double =
    tokenize(documentText).map(model.wordProbability(category, _)).sum + model.classProbability(category)
}

class Algorithm[K] {
  private var examples: List[(String, K)] = List()

  private val tokenize = (v: String) => v.split(' ')
  private val tokenizeTuple = (v: (String, K)) => tokenize(v._1)
  private val calculateWords = (l: List[(String, K)]) => l.map(tokenizeTuple(_).length).sum

  def addExample(ex: String, cl: K): Unit = examples ::= (ex, cl)

  def dictionary: Set[String] = examples.map(tokenizeTuple).flatten.toSet

  def model: Model[K] = {
    val docsByClass = examples.groupBy(_._2)
    val lengths = docsByClass.view.mapValues(calculateWords).toMap
    val docCounts = docsByClass.view.mapValues(_.length).toMap
    val wordsCount = docsByClass.view.mapValues(
      _.map(tokenizeTuple).flatten.groupBy(x => x).view.mapValues(_.length).toMap
    ).toMap

    new Model(lengths, docCounts, wordsCount, dictionary.size)
  }

  def classifier = new Classifier(model)
}
