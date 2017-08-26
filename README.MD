# jutsu.matrix

Use at your own risk! API subject to change.

You can install this library from clojars:
```clojure
[hswick/jutsu.matrix "0.0.7"]
```
This primarily serves as a light wrapper around nd4j linalg module and provides basic linear algebra support to those using ND4J arrays.
Names of functions can mostly assumed to be similar to the original nd4j library but with clojure naming conventions. This is mostly designed to help work with deeplearning4j with an added side bonus of a fast linear algebra library for clojure
that works on the CPU and GPU.

98% of the nd4j api has been wrapped, if for some reason you don't see a particular one feel free to submit a PR! :D

ND4J Resources

http://nd4j.org/doc/org/nd4j/linalg/ops/transforms/Transforms.html

http://nd4j.org/doc/org/nd4j/linalg/api/ndarray/BaseNDArray.html

http://nd4j.org/doc/org/nd4j/linalg/factory/Nd4j.html

# Usage

```clojure
(require '[jutsu.matrix.core :as jm])

;;This turns Clojure data into a matrix (NDArray type)
(def m1 (jm/matrix [[1 2 3 4] [1 2 3 4]]))

(def m2 (jm/matrix [[1 2 3 4] [1 2 3 4]]))

(println (jm/add m1 m2))
(println (jm/sub m1 m2))

(println (jm/transpose m1))

(println (jm/mmul m1 (jm/transpose m2)))

(jm/write-txt m1 "matrix1.txt")

(def m3 (jm/read-txt "matrix1.txt"))

(jm/equals? m3 m1)
;;=>true

;;Can create single row arrays with this syntax
(def m4 (jm/matrix 1 2 3 4))

(= (jm/mul m4 2) (jm/matrix [2 4 6 8]))
;;=> true

(= (jm/pow m4 2) (jm/matrix 1 4 9 16))
;;=> true

(jm/max-index (jm/matrix 1 2 3 4))
;;=>[4.0 3]

(jm/min-index (jm/matrix 1 2 3 4))
;;=>[1.0 0]

(jm/shape (jm/matrix [[1 2 3 4] [1 2 3 4]]))
;;=>[2 4]
```

# Why not core.matrix?

jutsu.matrix is inspired by the work of mikera and others on core.matrix. However, my desire to target the dl4j/nd4j ecosystem led me
astray from core.matrix for a couple of different reasons. The first is that attempts to write a core.matrix implementation never get the entire protocol
because nd4j does not support 0 dimension arrays. This means you can't actually swap out different implementations when using nd4j as a backend.

The second reason is that nd4j works as a protocol/interface because it has multiple backends for cpu and gpu. For me those are the only backends I need to switch out.
Personally, using multiple cpu libraries seems like added complexity. This means nd4j provides all the benefits of a protocol like core.matrix. 

Even though following the protocol is not a priority, users should find that they are very similar.

# Neanderthal

Yes, I have heard of neanderthal. jutsu.matrix has different priorities.

# FYI

Where names from nd4j conflict with clojure core names I usually attached an -array to the end
of the name to signify such an action was taking place on an NDArray.

For example reverse is reverse-array, or identity is identity-array.

One exception is concat which is concat-arrays because it works on multiple arrays.
# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

Run `boot test-code` to run your tests

## License

Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
