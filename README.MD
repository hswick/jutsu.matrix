# jutsu.matrix

Use at your own risk! API subject to change.

You can install this library from clojars:
```clojure
[hswick/jutsu.matrix "0.0.11"]
```

You will also need to include a ND4J backend for your dependencies as jutsu.matrix only wraps the ND4J api.

For CPU:
```clojure
[org.nd4j/nd4j-native-platform "0.8.0"]
```
For GPU: (CUDA must be installed)
```clojure
[org.nd4j/nd4j-cuda-8.0-platform "0.8.0"]
```

This primarily serves as a light wrapper around nd4j linalg module and provides basic linear algebra support to those using ND4J arrays.
Names of functions can mostly be assumed to be similar to the original nd4j library but with clojure naming conventions. The priority of this library is to facilitate work with deeplearning4j. Doesn't hurt that it is also a fast linear algebra library for Clojure that works on the CPU and GPU.

99% of the nd4j api has been wrapped, if for some reason you don't see a particular method feel free to submit a PR! :D

Currently need help with writing tests!

Here is a link to the [full documentation](https://hswick.github.io/jutsu.matrix/jutsu.matrix.core.html)

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

[full documentation](https://hswick.github.io/jutsu.matrix/jutsu.matrix.core.html)

# FYI

Where names from nd4j conflict with clojure core names I usually attached an -array to the end
of the name to signify such an action was taking place on an NDArray.

For example reverse is reverse-array, or identity is identity-array.

One exception is concat which is concat-arrays because it works on multiple arrays.

# Why not core.matrix?

jutsu.matrix is inspired by the work of mikera and others on core.matrix. However, my desire to target the dl4j/nd4j ecosystem led me
astray from core.matrix for a couple of different reasons. The first is that attempts to write a core.matrix implementation never get the entire protocol
because nd4j does not support 0 dimension arrays. This means you can't actually swap out different implementations when using nd4j as a backend.

The second reason is that nd4j is an interface with implemented and tested backends for both cpu and gpu. Simply, that is all I need to get my job done. Implementing another protocol on top of nd4j is redundant. IMO, using multiple different cpu libraries in one project seems like unnecessary complexity. Thus jutsu.matrix(nd4j) provides all the benefits of a protocol like core.matrix.

The third reason is that a lot of development on projects that implement core.matrix has been done. Unfortunately, at some point they have all run into a dead end. By removing the middle man I have been able to provide full coverage for the nd4j api in a shorter amount of time.

Even though following the core.matrix protocol is not a priority, users should find that jutsu.matrix feels very similar.

# Neanderthal

Yes, I have heard of neanderthal. jutsu.matrix has different priorities.

# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

Run `boot test-code` to run your tests

## License

Copyright © 2017 Harley Swick

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
