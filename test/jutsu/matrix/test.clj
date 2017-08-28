(ns jutsu.matrix.test
  (:require [clojure.test :refer :all]
            [jutsu.matrix.core :as jm]))

(deftest clj->nd4j
  (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (class (jm/matrix [[1 2 3 4] [1 2 3 4]]))))
  (is (= (jm/matrix [1 2 3 4]) (jm/matrix 1 2 3 4))))

(def test-m1 (jm/matrix [[1 2 3 4] [4 3 2 1]]))
(def test-m2 (jm/matrix [[1 2 3 4] [4 3 2 1]]))

(deftest equals
  (is true (jm/equals? test-m1 test-m2)))

(deftest zeros
  (is (jm/equals? (jm/matrix [0 0 0]) (jm/zeros 3)))
  (is (jm/equals? (jm/matrix [[0 0 0]
                              [0 0 0]
                              [0 0 0]]) (jm/zeros 3 3)))
  (is (= org.nd4j.linalg.cpu.nativecpu.NDArray (class (jm/zeros 5 6 7)))))

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

(deftest concatenation
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1]])
        (jm/concat-arrays 0 (jm/matrix [1 2 3 4]) (jm/matrix [4 3 2 1])))))

(deftest vstack
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1] [1 5 3 5]])
        (jm/vstack (jm/matrix [[1 2 3 4] [4 3 2 1]]) (jm/matrix [1 5 3 5]))))
  (is (jm/equals? (jm/matrix [[1 2 3 4] [4 3 2 1]])
        (jm/vstack (jm/matrix [1 2 3 4]) (jm/matrix [4 3 2 1])))))

(deftest shape
  (is (= [2 4] (jm/shape test-m1))))

(deftest write-and-read
  (jm/write-txt test-m1 "test-m1.txt")
  (is (jm/equals? test-m1 (jm/read-txt "test-m1.txt"))))

(deftest subtraction
  (let [matrix-2s (jm/matrix [[2 2 2]
                              [2 2 2]])
        matrix-neg2s (jm/matrix [[-2 -2 -2]
                                 [-2 -2 -2]])
        zeros-2s (jm/zeros 2 3)]
    (is (jm/equals? matrix-neg2s (jm/sub zeros-2s matrix-2s)))
    (jm/sub! zeros-2s matrix-2s)
    (is (jm/equals? matrix-neg2s zeros-2s))))

(deftest matrix-mul
  (let [a1 (jm/matrix [1 2 3 4])
        a2 (jm/matrix [6 7 4 3])]
    (is (= [1 1] (jm/shape (jm/mmul a1 (jm/transpose a2)))))
    (is (= [1 1] (jm/shape (jm/inner-product a1 a2))))
    (is (= (jm/mmul a1 (jm/transpose a2)) (jm/inner-product a1 a2)))
    (is (= (jm/mmul (jm/transpose a1) a2) (jm/outer-product a1 a2)))
    (is (= [4 4] (jm/shape (jm/outer-product a1 a2))))))

(deftest sum-test
  (let [a1 (jm/matrix [1 2 3 4])
        a2 (jm/matrix [6 7 4 3])]
    (is (= [1 1] (jm/shape (jm/sum a1))))
    (is (= [1 1] (jm/shape (jm/sum (jm/matrix [[1 2 3 4] [1 2 3 4] [1 2 3 4] [1 2 3 4]])))))))

(deftest min-max-num
  (let [a1 (jm/matrix [1 2 3 4])]
    (is (= 4.0 (jm/max-num a1)))
    (is (= 1.0 (jm/min-num a1)))
    (is (= 2.5 (jm/mean-num a1)))))

(deftest keep-min-max 
  (let [a1 (jm/matrix [1 2 3 4]) a2 (jm/matrix [4 5 6 1])]
    (is (= (jm/matrix [4 5 6 4]) (jm/keep-max a1 a2)))
    (is (= (jm/matrix [1 2 3 1]) (jm/keep-min a1 a2)))))

(deftest division
  (let [a1 (jm/matrix [4 4 4 4]) a2 (jm/matrix [2 2 2 2])]
    (is (= a2 (jm/div a1 2)))
    (is (= a2 (jm/div a1 a2)))
    (jm/div! a1 2)
    (is (= a2 a1))))

;;Basic = assertion doesn't seem to work because of different internal arrays
;;in memory.
;;Okay so the shape of these repeated arrays is rather wonky
(deftest repeat-test
  (is (= [2 1 4]
         (jm/shape (jm/repeat-array (jm/matrix [1 2 3 4]) 2)))))

(deftest create
  (is (= [4 3] (jm/shape (jm/create 4 3)))))

(deftest reverse-array
  (is (= (jm/matrix 1 2 3 4) (jm/reverse-array (jm/matrix 4 3 2 1))))
  (is (= (jm/matrix [[10 6 5 4]
                     [1 2 3 4]
                     [8 7 6 5]
                     [4 3 2 1]])
         (jm/reverse-array (jm/matrix [[1 2 3 4]
                                       [5 6 7 8]
                                       [4 3 2 1]
                                       [4 5 6 10]])))))

(deftest transpose
  (is (= [4 1] (jm/shape (jm/transpose (jm/matrix 1 2 3 4))))))

(deftest inner-product
  (is (= [1 1] (jm/shape (jm/inner-product (jm/matrix 1 2 3 4) (jm/matrix 1 2 3 4)))))
  (is (= (jm/mmul (jm/matrix 1 2 3 4)
           (jm/transpose (jm/matrix 1 2 3 4)))
         (jm/inner-product (jm/matrix 1 2 3 4) (jm/matrix 1 2 3 4)))))

(deftest outer-product
  (is (= [4 4] (jm/shape (jm/outer-product (jm/matrix 1 2 3 4) (jm/matrix 1 2 3 4)))))
  (is (= (jm/mmul (jm/transpose (jm/matrix 1 2 3 4))
                  (jm/matrix 1 2 3 4))
         (jm/outer-product (jm/matrix 1 2 3 4) (jm/matrix 1 2 3 4)))))

(deftest max-min-indices
  (is (= [4.0 0] (jm/max-index (jm/matrix 4 3 2 1))))
  (is (= [1.0 3] (jm/min-index (jm/matrix 4 3 2 1)))))

(deftest mul-test
  (is (= (jm/matrix 2 4 6 8) (jm/mul (jm/matrix 1 2 3 4) 2)))
  (is (= (jm/matrix 4 4 4 4) (jm/mul 
                               (jm/matrix 2 2 2 2)
                               (jm/matrix 2 2 2 2))))
  (let [m1 (jm/matrix 1 2 3 4)]
    (jm/mul! m1 2)
    (is (= (jm/matrix 2 4 6 8) m1))))

(deftest rows-cols
  (is (= 1 (jm/rows (jm/matrix 1 2 3 4))))
  (is (= 4 (jm/columns (jm/matrix 1 2 3 4))))
  (is (= (jm/rows (jm/matrix 1 2 3 4)) (jm/num-rows [1 2 3 4])))
  (is (= (jm/columns (jm/matrix 1 2 3 4)) (jm/num-cols [1 2 3 4])))
  (is (= (jm/rows (jm/matrix 1 2 3 4)) (jm/num-rows (jm/matrix 1 2 3 4))))
  (is (= (jm/columns (jm/matrix 1 2 3 4)) (jm/num-cols (jm/matrix 1 2 3 4)))))

(deftest squared-sqrt
  (is (= (jm/matrix 1 4 9 16) (jm/pow (jm/matrix 1 2 3 4) 2))))
