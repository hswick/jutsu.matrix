(ns jutsu.matrix.test
  (:require [clojure.test :refer :all]
            [jutsu.matrix.core :as jm]))

(println (jm/matrix [[1 2 3 4] [1 2 3 4]]))

(println (jm/add
           (jm/matrix [[1 2 3] [1 2 3]])
           (jm/matrix [[3 2 1] [3 2 1]])))

(println (jm/to-string (jm/matrix [1 3 4])))

(println (jm/add!
           (jm/matrix [[1 2 3] [1 2 3]])
           (jm/matrix [[3 2 1] [3 2 1]])))

(println (jm/add! (jm/zeros 1 4) (jm/matrix [1 2 3 4])))
