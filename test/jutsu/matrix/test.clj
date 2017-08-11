(ns jutsu.matrix.test
  (:require [clojure.test :refer :all]
            [jutsu.matrix.core :as jm]))

(deftest clj->nd4j
  (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (class (jm/matrix [[1 2 3 4] [1 2 3 4]])))))

(def test-m1 (jm/matrix [[1 2 3 4] [4 3 2 1]]))
(def test-m2 (jm/matrix [[1 2 3 4] [4 3 2 1]]))

(deftest equals
  (is true (jm/equals? test-m1 test-m2)))

(deftest zeros
  (is (jm/equals? (jm/matrix [0 0 0]) (jm/zeros 3)))
  (is (jm/equals? (jm/matrix [[0 0 0]
                              [0 0 0]
                              [0 0 0]]) (jm/zeros 3 3))))

(deftest ones
  (is (jm/equals? (jm/matrix [1 1 1]) (jm/ones 3)))
  (is (jm/equals? (jm/matrix [[1 1 1]
                              [1 1 1]
                              [1 1 1]]) (jm/ones 3 3))))

(deftest addition
  (let [matrix-2s (jm/matrix [[2 2 2]
                              [2 2 2]])
        zeros-2s (jm/zeros 2 3)]
    (is (jm/equals? matrix-2s (jm/add zeros-2s matrix-2s)))
    (jm/add! zeros-2s matrix-2s)
    (is (jm/equals? matrix-2s zeros-2s))))

(deftest to-string
  (is (= "[1.00, 3.00, 4.00]" (jm/to-string (jm/matrix [1 3 4])))))

(deftest concat
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1]])
        (jm/concat 0 (jm/matrix [1 2 3 4]) (jm/matrix [4 3 2 1])))))

(deftest vstack
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1] [1 5 3 5]])
        (jm/vstack (jm/matrix [[1 2 3 4] [4 3 2 1]]) (jm/matrix [1 5 3 5]))))
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1]])
        (jm/vstack (jm/matrix [1 2 3 4]) (jm/matrix [4 3 2 1])))))


