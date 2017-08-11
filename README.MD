# jutsu.matrix

Use at your own risk! API subject to change.

You can install this library from clojars:
```clojure
[hswick/jutsu.matrix "0.0.4"]
```
This primarily serves as a light wrapper around nd4j linalg module and provides basic linear algebra support to those using ND4J arrays.
Names of functions can be assumed to be similar to the original nd4j library but with clojure naming conventions.

Most functions have been implemented. If for some reason you don't see a particular one feel free to submit a PR! :D

# Usage

```clojure
(require '[jutsu.matrix.core :as jm])

;;This turns Clojure data into 
(def m1 (jm/matrix [[1 2 3 4] [1 2 3 4]]))

(def m2 (jm/matrix [[1 2 3 4] [1 2 3 4]]))

(println (jm/add m1 m2))
(println (jm/sub m1 m2))

(println (jm/transpose m1))

(println (jm/mmul m1 (jm/transpose m2)))
```

# Why not core.matrix?

jutsu.matrix is inspired by the work of mikera and others on core.matrix. However, my desire to target the dl4j/nd4j ecosystem led me
astray from core.matrix for a couple of different reasons. The first is that attempts to write a core.matrix implementation never get the entire protocol
because nd4j does not support 0 dimension arrays. This means you can't actually swap out different implementations when using nd4j as a backend.

The second reason is that nd4j works as a protocol/interface because it has multiple backends for cpu and gpu. For me those are the only backends I need to switch out.
Personally, using multiple cpu libraries seems like added complexity. This means nd4j provides all the benefits of a protocol like core.matrix. 

Thus means following the core.matrix protocol is not a priority of mine. 
# Dev

Run `boot night` to startup nightlight and begin editing your project in a browser.

Run `boot test-code` to run your tests

## License

Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
