(ns jutsu.matrix.core
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.ops.transforms Transforms]
           [org.nd4j.linalg.api.ndarray INDArray]))

;;http://nd4j.org/doc/org/nd4j/linalg/ops/transforms/Transforms.html
;;http://nd4j.org/doc/org/nd4j/linalg/api/ndarray/BaseNDArray.html
;;http://nd4j.org/doc/org/nd4j/linalg/factory/Nd4j.html

(defn num-rows
  "Get number of rows in ndarray or clojure coll."
  [coll]
  (if (instance? INDArray coll)
    (first (.shape coll))
    (if (coll? (first coll)) (count coll) 1)))

(defn num-cols
  "Get number of columns in ndarray or clojure coll."
  [coll]
  (if (instance? INDArray coll)
    (second (.shape coll))
    (if (coll? (first coll)) (count (first coll)) (count coll))))

(defn matrix 
  "Converts clojure data to a ND4J matrix or vector."
  ([coll]
   (let [h (num-rows coll) w (num-cols coll)]
    (if (= h 1)
      (Nd4j/create (float-array (seq coll)))
      (let [new-array (Nd4j/create h w)]
        (doseq [i (range 0 h)]
          (.putRow new-array i (Nd4j/create (float-array (nth coll i)))))
        new-array))))
  ([n & args] (matrix (cons n args))))

(defn create
  "Create a new empty array with given shape."
  [rows cols]
  (Nd4j/create rows cols))

;;ND4J static methods
(defn zeros
  "Returns an array full of zeros based on number(s) given"
  ([cols] (Nd4j/zeros cols))
  ([rows cols] (Nd4j/zeros rows cols)))

(defn diag
  "Creates a new matrix where the values of the given vector are the diagonal values of the matrix if a vector is passed in, if a matrix is returns the kth diagonal in the matrix."
  ([ndarray] (Nd4j/diag ndarray))
  ([ndarray k] (Nd4j/diag ndarray k)))

(defn from-byte-array
  "Read an ndarray from a byte array."
  [byte-array] (Nd4j/fromByteArray byte-array))

(defn clear-nans 
  "Clear nans from an ndarray."
  [ndarray] (Nd4j/clearNans ndarray))

(defn get-backend
  "Get information about current backend like CPU vs GPU."
  [] (Nd4j/getBackend))

(defn get-compressor
  "This method returns BasicNDArrayCompressor instance, suitable for NDArray compression/decompression at runtime."
  [] (Nd4j/getCompressor))

(defn get-fft
  "Returns the fft instance."
  [] (Nd4j/getFFt))

(defn fallback-mode-enabled?
  "Checks if fallback mode was enabled."
  [] (Nd4j/isFallbackModeEnabled))

(defn ones
  "Returns an array full of ones based on number(s) given"
  ([cols] (Nd4j/ones cols))
  ([rows cols] (Nd4j/ones rows cols)))

(defn prod 
  ([ndarray] (Nd4j/prod ndarray))
  ([ndarray k] (Nd4j/prod ndarray k)))

(defn rand-array
  "Create a random ndarray with the given shape using the current time as the seed."
  ([shape] (Nd4j/rand shape))
  ([rows cols] (Nd4j/rand cols))
  ([rows cols seed] (Nd4j/rand cols))
  ([rows cols min max rng] (Nd4j/rand cols min max rng)))

(defn randn-array
  "Random normal using the current time stamp as the seed"
  ([shape] (Nd4j/randn shape))
  ([rows columns] (Nd4j/randn rows columns))
  ([rows columns rng] (Nd4j/randn rows columns rng)))

(defn repeat-array
  "Repeats the input array n times to create a new larger array."
  [ndarray n] (Nd4j/repeat ndarray n))

(defn roll-axis
  "Roll the specified axis backwards, until it lies in a given position."
  ([ndarray axis] (Nd4j/rollAxis ndarray axis))
  ([ndarray axis start] (Nd4j/rollAxis ndarray axis start)))

(defn rotate
 "Reverses the passed in matrix such that m[0] becomes m[m.length - 1] etc."
 [ndarray] (Nd4j/rot ndarray))

(defn rotate-90
  "Rotate a matrix 90 degrees"
  [ndarray] (Nd4j/rot90 ndarray))

(defn scalar
  "Create a scalar ndarray with the specified value and/or offset"
  ([value] (Nd4j/scalar value))
  ([value offset] (Nd4j/scalar value offset)))

(defn sort-array
  "Sort an ndarray along a particular dimension."
  [ndarray dimension ascending] (Nd4j/sort ndarray dimension ascending))

(defn sort-columns
  "Sort (shuffle) the columns of a 2d array according to the value at a specified row."
  [ndarray row-idx ascending] (Nd4j/sortColumns ndarray row-idx ascending))

(defn sort-rows
  "Sort (shuffle) the rows of a 2d array according to the value at a specified column."
  [ndarray col-idx ascending] (Nd4j/sortRows ndarray col-idx ascending))

(defn sort-with-indices
  "Sort an ndarray along a particular dimension and also return the sorted indices."
  [ndarray dimension ascending] (Nd4j/sortWithIndices ndarray dimension ascending))

(defn sum
  "[ndarray] Sums all numbers in array even if multiple rows."
  ([ndarray] (Nd4j/sum ndarray))
  ([ndarray dimension] (Nd4j/sum ndarray dimension)))

(defn to-byte-array
  "Convert an ndarray to a byte array."
  [ndarray] (Nd4j/toByteArray ndarray))

(defn to-flattened
  "Create a long row vector of all of the given ndarrays."
  [ndarrays] (Nd4j/toFlattened ndarrays))

(defn value-array-of
  "Creates an ndarray with the specified value as the only value in the ndarray."
  ([shape value] (Nd4j/valueArrayOf shape value))
  ([rows columns value] (Nd4j/valueArrayOf rows columns value)))

(defn var
  ([ndarray] (Nd4j/var ndarray))
  ([ndarray dimension] (Nd4j/var ndarray dimension)))

(defn allows-specify-ordering?
  "Backend specific: Returns whether specifying the order for the blas impl is allowed (cblas)."
  []
  (Nd4j/allowsSpecifyOrdering))

(defn append
  "Append the given array with the specified value size along a particular axis."
  [arr pad-amount val axis]
  (Nd4j/append arr pad-amount val axis))

(defn arange
  "Array of evenly spaced values."
  ([end] (Nd4j/arange end))
  ([begin end] (Nd4j/arange begin end)))

(defn average-and-propagate
  "This method averages input arrays, and returns averaged array."
  ([arrays] (Nd4j/averageAndPropagate arrays))
  ([target arrays] (Nd4j/averageAndPropagate target arrays)))

(defn bilinear-products
  "Returns a column vector where each entry is the nth bilinear product of the nth slices of the two tensors."
  [curr in]
  (Nd4j/bilinearProducts curr in))

(defn buffer-ref-queue
  "The reference queue used for cleaning up databuffers."
  []
  (Nd4j/bufferRefQueue))

(defn choice
  "This method samples value from Source array to Target, with probabilites provided in Probs argument."
  ([source probs num-samples] (Nd4j/choice source probs num-samples))
  ([source probs num-samples rng] (Nd4j/choice source probs num-samples rng)))

(defn concat-arrays
  "Concatenate arrays along a dimension."
  [dimension & args] (Nd4j/concat dimension (into-array args)))

(defn data-type
  "Returns the data type used for the runtime."
  [] (Nd4j/dataType))

(defn empty-like
  "Empty like."
  [arr] (Nd4j/emptyLike arr))

(defn enable-fallback-mode!
  "Enables fallback to safe-mode for specific operations."
  [really-enable] (Nd4j/enableFallbackMode really-enable))

(defn eye
  "Create the identity ndarray."
  [n] (Nd4j/eye n))

(defn factory
  "The factory used for creating ndarrays."
  [] (Nd4j/factory))

(defn get-affinity-manager [] (Nd4j/getAffinityManager))

(defn get-constnat-handler [] (Nd4j/getConstantHandler))

(defn get-convolution
  "Get the convolution singleton."
  [] (Nd4j/getConvolution))

(defn get-data-buffer-factory [] (Nd4j/getDataBufferFactory))

(defn get-distributions
  "Get the primary distributions factory."
  [] (Nd4j/getDistributions))

(defn get-instrumentation
  "Gets the instrumentation instance."
  [] (Nd4j/getInstrumentation))

(defn get-memory-manager
  "This method returns backend-specific MemoryManager implementation, for low-level memory management."
  [] (Nd4j/getMemoryManager))

(defn get-ndarray-factory [] (Nd4j/getNDArrayFactory))

(defn get-op-factory
  "Get the operation factory."
  [] (Nd4j/getOpFactory))

(defn get-random-generator
  "Get the current random generator"
  [] (Nd4j/getRandom))

(defn get-random-factory
  "This method returns RandomFactory instance."
  [] (Nd4j/getRandomFactory))

(defn get-strides
  "Get the strides based on the shape and NDArrays.order()"
  ([shape] (Nd4j/getStrides shape))
  ([shape order] (Nd4j/getStrides shape order)))

(defn hstack
  "Concatenates two matrices horizontally."
  ([arr & args] (Nd4j/hstack (into-array args))))

(defn ones-like
  "Ones like."
  [arr] (Nd4j/onesLike arr))

(defn pad
  "Pad the given ndarray to the size along each dimension."
  ([to-pad pad-width pad-mode] (Nd4j/pad to-pad pad-width pad-mode))
  ([to-pad pad-width constant-values pad-mode]
   (Nd4j/pad to-pad pad-width constant-values pad-mode)))

(defn prepend
  "Append the given array with the specified value size along a particular axis."
  [arr pad-amount val axis] (Nd4j/prepend arr pad-amount val axis))

(defn pull-rows
  "This method produces concatenated array, that consist from tensors, fetched from source array, against some dimension and specified indexes."
  ([source source-dim indexes] (Nd4j/pullRows source source-dim indexes))
  ([source source-dim indexes order] (Nd4j/pullRows source source-dim indexes order)))

(defn read-binary
  "Read a binary ndarray from the given file."
  [read-file] (Nd4j/readBinary read-file))

(defn read-array
  "Read in an ndarray from a data input stream."
  [input] (Nd4j/read input))

(defn read-numpy
  "Create a ndarray by reading in a serialized numpy array."
  ([file-path] (Nd4j/readNumpy file-path))
  ([file-path split] (Nd4j/readNumpy file-path split)))

(defn read-txt
  "Create a ndarray by reading in a text file."
  ([file-path] (Nd4j/readTxt file-path))
  ([file-path sep] (Nd4j/readTxt file-path sep)))

(defn read-txt-string
  "Create ndarray by parsing a string."
  ([ndarray] (Nd4j/readTxtString ndarray))
  ([ndarray sep] (Nd4j/readTxtString ndarray sep)))

(defn ref-queue
  "The reference queue used for cleaning up ndarrays."
  [] (Nd4j/refQueue))

(defn reverse-array
  "Reverses elements in each row and reverses the order of rows."
  [arr] (Nd4j/reverse arr))

(defn save-binary
  "Save an ndarray to the given file."
  [arr save-to] (Nd4j/saveBinary arr save-to))

(defn set-data-type!
  "This method sets dataType for the current JVM runtime."
  [dtype] (Nd4j/setDataType dtype))

(defn shape
  "Returns the shape of the ndarray."
  [arr] 
  (into [] (Nd4j/shape arr)))

(defn shuffle-array!
  "Symmetric in place shuffle of an ndarray along a specified set of dimensions."
  ([to-shuffle & args] (Nd4j/shuffle to-shuffle (into-array args))))

(defn size-of-data-type
  "This method returns sizeOf(currentDataType), in bytes."
  []
  (Nd4j/sizeOfDataType))

(defn vstack
  "Concatenates two matrices vertically."
  ([& args] (Nd4j/vstack (into-array args))))

(defn write
  "Write an ndarray to the specified outputstream."
  [arr output-stream] (Nd4j/write arr output-stream))

(defn write-numpy
  "Write ndarray to a file in numpy form."
  [write file-path split] (Nd4j/writeNumpy write file-path split))

(defn write-txt
  "Write ndarray to text file." 
  ([write file-path] (Nd4j/writeTxt write file-path))
  ([write file-path split] (Nd4j/writeTxt write file-path split))
  ([write file-path split precision] (Nd4j/writeTxt write file-path split precision)))

(defn write-txt-string
  "Write ndarray as a string to an output stream."
  ([write os] (Nd4j/writeTxtString write os))
  ([write os split] (Nd4j/writeTxtString write os split))
  ([write os split precision] (Nd4j/writeTxtString write os split precision)))

(defn zeros-like
  "Zeros like."
  [arr] (Nd4j/zerosLike arr))

;;Transform static methods
(defn abs
  "Take the absolute of the array and return a new array."
  [ndarray] (Transforms/abs ndarray true))

(defn abs!
  "Take the absolute of the array and change in place."
  [ndarray] (Transforms/abs ndarray false))

(defn sigmoid
  "Pass ndarray through sigmoid function and return a new array."
  [ndarray] (Transforms/sigmoid ndarray true))

(defn sigmoid!
  "Pass ndarray through sigmoid function and change in place."
  [ndarray] (Transforms/sigmoid ndarray false))

(defn tanh
  "Pass ndarray through tanh function and return a new array."
  [ndarray] (Transforms/tanh ndarray true))

(defn tanh!
  "Pass ndarray through tanh function and change in place."
  [ndarray] (Transforms/tanh ndarray false))

(defn sqrt
  "Pass ndarray through square root function and return new array."
  [ndarray] (Transforms/sqrt ndarray true))

(defn sqrt!
  "Pass ndarray through square root function and change in place."
  [ndarray] (Transforms/sqrt ndarray false))

(defn exp 
  "Square euler's number by the array and return new array."
  [ndarray] (Transforms/exp ndarray true))

(defn exp!
  "Square euler's number by the array and change in place."
  [ndarray] (Transforms/exp ndarray false))

(defn and-arrays
  "Return new array that is result of element wise AND comparison of both arrays."
  [x y] (Transforms/and x y))

(defn acos
  "Pass ndarray through arccosine function and return new array."
  [ndarray] (Transforms/acos ndarray true))

(defn acos!
  "Pass ndarray through arccosine function and change in place."
  [ndarray] (Transforms/acos ndarray false))

(defn asin
  "Pass ndarray through arcsine function and return new array."
  [ndarray] (Transforms/asin ndarray true))

(defn asin!
  "Pass ndarray through arcsine function and change in place."
  [ndarray] (Transforms/asin ndarray false))

(defn atan
  "Pass ndarray through arctangent function and return new array."
  [ndarray] (Transforms/atan ndarray true))

(defn atan!
  "Pass ndarray through arctangent function and change in place."
  [ndarray] (Transforms/atan ndarray false))

(defn ceil
  "Rounds all numbers up and returns new array."
  [ndarray] (Transforms/ceil ndarray true))

(defn ceil!
  "Rounds all numbers up and changes in place."
  [ndarray] (Transforms/ceil ndarray false))

(defn ceiling
  "Returns binary matrix of whether the number at a given index is greater than"
  [ndarray] (Transforms/ceiling ndarray true))

(defn ceiling!
  "Changes array in place to create binary matrix of whether the number at a given index is greater than"
  [ndarray] (Transforms/cos ndarray false))

(defn cos
  "Pass ndarray through cosine function and return new array."
  [ndarray] (Transforms/ceiling ndarray true))

(defn cos!
  "Pass ndarray through cosine function and change in place."
  [ndarray] (Transforms/cos ndarray false))

(defn cosine-sim
  "Get the cosine similarity between two arrays."
  [ndarray ndarray2] (Transforms/cosineSim ndarray ndarray2))

(defn eps
  "Pass ndarray through eps function and return new array."
  [ndarray] (Transforms/eps ndarray true))

(defn eps!
  "pass array through eps function and change in place."
  [ndarray] (Transforms/eps ndarray false))

(defn floor
  "Rounds all values in the array down and returns new array."
  [ndarray] (Transforms/floor ndarray true))

(defn floor! 
  "Rounds all values in the array down and changes in place."
  [ndarray] (Transforms/floor ndarray false))

(defn greater-than-or-equal
  "1 if greater than or equal to 0 otherwise (at each element), like eps function and returns new array."
  [ndarray ndarray2] (Transforms/greaterThanOrEqual ndarray ndarray2 true))

(defn greater-than-or-equal!
  "1 if greater than or equal to 0 otherwise (at each element), like eps function and changes first array in place."
  [ndarray ndarray2] (Transforms/greaterThanOrEqual ndarray ndarray2 false))

(defn hard-tanh
  "Passes ndarray through hard tanh function and returns new array."
  [ndarray] (Transforms/hardTanh ndarray true))

(defn hard-tanh!
  "Passes ndarray through hard tanh function and changes in place."
  [ndarray] (Transforms/hardTanh ndarray false))

(defn identity-array
  "Passes ndarray through identity function and returns new array."
  [ndarray] (Transforms/identity ndarray true))

(defn identity-array! [ndarray]
  "Passes ndarray through identity function and changes in place."
  (Transforms/identity ndarray false))

(defn leakyRelu [ndarray]
  "Passes ndarray through leaky relu function and returns new array."
  (Transforms/leakyRelu ndarray true))

(defn leakyRelu!
  "Passes ndarray through leaky relu function and changes in place."
  [ndarray] (Transforms/leakyRelu ndarray false))

(defn less-than-or-equal
  "1 if less than or equal to 0 otherwise (at each element) and returns new array"
  [ndarray ndarray2] (Transforms/lessThanOrEqual ndarray ndarray2 true))

(defn less-than-or-equal!
  "1 if less than or equal to 0 otherwise (at each element) and changes first array in place"
  [ndarray ndarray2] (Transforms/lessThanOrEqual ndarray ndarray2 false))

(defn log
  "Pass ndarray through log function (optionally choosing base) and return new array."
  ([ndarray] (Transforms/log ndarray true))
  ([ndarray base] (Transforms/log ndarray base true)))

(defn log!
  "Pass ndarray through log function (optionally choosing base) and change in place."
  ([ndarray] (Transforms/log ndarray false))
  ([ndarray base] (Transforms/log ndarray base false)))

(defn manhattan-distance
  "Get the manhattan distance between two arrays."
  [ndarray ndarray2]
  (Transforms/manhattanDistance ndarray ndarray2))

(defn keep-max
  "Element wise maximum function between two arrays or a single array and a value, returns new array."
  ([ndarray k] (Transforms/max ndarray k true)))

(defn keep-max!
  "Element wise maximum function between two arrays or a single array and a value, changes in place."
  [ndarray k] (Transforms/max ndarray k false))

(defn keep-min
  "Element wise minimum function between two arrays or a single array and a value, returns new array."
  [ndarray k] (Transforms/min ndarray k true))

(defn keep-min!
  "Element wise minimum function between two arrays or a single array and a value, changes in place."
  [ndarray k] (Transforms/min ndarray k false))

(defn neg
  "Return negative of array."
  [ndarray] (Transforms/neg ndarray true))

(defn neg!
  "Change array to its negative form"
  [ndarray] (Transforms/neg ndarray false))

(defn normalize!
  "Normalize data to zero mean and unit variance substract by the mean and divide by the standard deviation."
  [ndarray] (Transforms/normalizeZeroMeanAndUnitVariance ndarray))

(defn not-array
  "Perform element wise not on array and return new array"
  [ndarray] (Transforms/not ndarray))

(defn or-arrays
  "Perform element wise or comparison of two arrays."
  [ndarray ndarray2] (Transforms/or ndarray ndarray2))

(defn pow
  "Pass ndarray through power function (specifying the power) and returns new array."
  [ndarray power] (Transforms/pow ndarray power true))

(defn pow!
  "Pass ndarray through power function (specifying the power) and change in place."
  [ndarray power] (Transforms/pow ndarray power false))

(defn relu 
  "Pass ndarray through relu function and return new array. Very popular for machine learning applications."
  [ndarray] (Transforms/relu ndarray true))

(defn relu!
  "Pass ndarray through relu function and change in place. Very popular for machine learning applications."
  [ndarray] (Transforms/relu ndarray false))

(defn round
  "Pass ndarray through rounding function and return new array."
  [ndarray] (Transforms/round ndarray true))

(defn round!
  "Pass ndarray through rounding function and change in place."
  [ndarray] (Transforms/round ndarray false))

(defn sign
  "Pass ndarray through signum function and return new array."
  [ndarray] (Transforms/sign ndarray true))

(defn sign!
  "Pass ndarray through signum function and change in place."
  [ndarray] (Transforms/sign ndarray false))

(defn sin
  "Pass ndarray through sine function and return new array."
  [ndarray] (Transforms/sin ndarray true))

(defn sin!
  "Pass ndarray through sine function and change in place."
  [ndarray] (Transforms/sin ndarray false))

(defn soft-plus
  "Pass ndarray through soft plus function and return new array."
  [ndarray] (Transforms/softPlus ndarray true))

(defn soft-plus! 
  "Pass ndarray through soft plus and change in place."
  [ndarray] (Transforms/softPlus ndarray false))

(defn stabilize
  "Stabilize to be within a range of k and return new array."
  [ndarray k] (Transforms/stabilize ndarray k true))

(defn stabilize! 
  "Stabilize to be within a range of k and change in place."
  [ndarray k] (Transforms/stabilize ndarray k false))

(defn unit-vec
  "Scale by 1 / norm2 of the matrix."
  [ndarray] (Transforms/unitVec ndarray))

(defn xor
  "Element wise xor comparison on two arrays."
  [ndarray ndarray2] (Transforms/xor ndarray ndarray2))

;;ND4j Methods
(defn linear-view
  "Returns a linear view reference of shape 1,length(ndarray)."
  [ndarray] (.linearView ndarray))

(defn transpose
  "Return transposed copy of ndarray."
  [ndarray] (.transpose ndarray))

(defn transpose!
  "Return transposed version of ndarray and change in place"
  [ndarray] (.transposei ndarray))

(defn mmul
  "Perform matrix multiplication and return new array."
  [ndarray ndarray2] (.mmul ndarray ndarray2))

(defn mmul!
  "Perform matrix multiplication and change first array in place."
  [ndarray ndarray2] (.mmuli ndarray ndarray2))

(defn mul-column-vector
  "Multiplication of a column vector, returns new array."
  [ndarray column-vector] (.mulColumnVector ndarray column-vector))

(defn mul-column-vector!
  "Multiplication of a column vector, changes in place."
  [ndarray column-vector] (.muliColumnVector ndarray column-vector))

(defn mul-row-vector
  "Multiplication of a row vector, returns new array."
  [ndarray row-vector] (.mulRowVector ndarray row-vector))

(defn mul-row-vector!
  "Multiplication of a row vector, changes in place."
  [ndarray row-vector] (.muliRowVector ndarray row-vector))

(defn reshape
  "Reshapes the ndarray (can't change the length of the ndarray), changes in place."
  [ndarray rows cols] (.reshape ndarray rows cols))

(defn add
  "Addition of two arrays, returns new array"
  [ndarray ndarray2] (.add ndarray ndarray2))

(defn add!
  "Addition of two arrays, change in place."
  [ndarray ndarray2] (.addi ndarray ndarray2))

(defn add-column-vector
  "Addition of a column vector, returns new array."
  [ndarray column-vector]
  (.addColumnVector ndarray column-vector))

(defn add-column-vector!
  "Addition of a column vector, changes in place."
  [ndarray column-vector]
  (.addiColumnVector ndarray column-vector))

(defn add-row-vector
  "Addition of a row vector, returns new array."
  [ndarray row-vector]
  (.addRowVector ndarray row-vector))

(defn add-row-vector!
  "Addition of a row vector, changes in place."
  [ndarray row-vector]
  (.addiRowVector ndarray row-vector))

(defn assign
  "Assign all of the elements in the given ndarray to this ndarray or value."
  [ndarray1 ndarray2] (.assign ndarray1 ndarray2))

(defn check-dimensions
  "Validate dimensions are equal."
  [ndarray ndarray2] (.checkDimensions ndarray ndarray2))

(defn cleanup!
  "Cleanup resources."
  [ndarray] (.cleanup ndarray))

(defn columns
  "Number of columns (shape[1]), throws an exception when called when not 2d."
  [ndarray] (.columns ndarray))

(defn cumsum
  "Cumulative sum along a dimension (in place), returns new array."
  [ndarray dimension] (.cumsum ndarray dimension))

(defn cumsum!
  "Cumulative sum along a dimension (in place), chanes in place."
  [ndarray dimension] (.cumsumi ndarray dimension))

(defn get-data
  "Returns a linear double array representation of this ndarray"
  [ndarray] (.data ndarray))

(defn one-norm-distance
  "Returns the (1-norm) distance."
  [ndarray ndarray2] (.distance1 ndarray ndarray2))

(defn div
  "Divide two arrays and return new array."
  [ndarray ndarray2] (.div ndarray ndarray2))

(defn div!
  "Divide two arrays and change first array in place."
  [ndarray ndarray2] (.divi ndarray ndarray2))

(defn div-column-vector
  "Division of array by column vector, returns new array."
  [ndarray column-vector] (.divColumnVector ndarray column-vector))

(defn div-column-vector!
  "Division of array by column vector, changes in place."
  [ndarray column-vector] (.diviColumnVector ndarray column-vector))

(defn div-row-vector
  "Division of array by row vector, returns new array."
  [ndarray row-vector] (.divRowVector ndarray row-vector))

(defn div-row-vector!
  "Division of array by row vector, changes in place."
  [ndarray row-vector] (.diviRowVector ndarray row-vector))

(defn dup
  "Returns a copy of this ndarray, (optionally) where the returned ndarray has the specified order."
  ([ndarray] (.dup ndarray))
  ([ndarray order] (.dup ndarray order)))

(defn element-stride
  "Element stride (one element to the next, also called the defualt stride: 1 for normal 2 for complex)."
  [ndarray] (.elementStride ndarray))

(defn element-wise-stride
  "Element wise stride."
  [ndarray] (.element-wise-stride ndarray))

(defn eq
  "Returns the binary ndarray for 'Equals' comparison."
  [ndarray ndarray2] (.eq ndarray ndarray2))

(defn eq!
  "Returns the binary ndarray for 'Equals' comparison, changes in place."
  [ndarray ndarray2] (.eqi ndarray ndarray2))

(defn equals?
  "Compare two matrices."
  [ndarray ndarray2] (.equals ndarray ndarray2))

(defn equals-with-eps?
  "This method allows you to compare INDArray against other INDArray, with variable eps."
  [ndarray ndarray2 eps-value] (.equalsWithEps ndarray ndarray2 eps-value))

(defn fmod
  "Remainder of division returns new array."
  [ndarray ndarray2] (.fmod ndarray ndarray2))

(defn fmod!
  "Remainder of division changes in place."
  [ndarray ndarray2] (.fmodi ndarray ndarray2))

(defn get-column
  "Get the specified column."
  [ndarray c] (.getColumn ndarray c))

(defn get-double
  "Returns the elements at the specified indices"
  ([ndarray i] (.getDouble ndarray i))
  ([ndarray i j] (.getDouble ndarray i j)))

(defn get-float
  "Returns the elements at the specified indices."
  ([ndarray i] (.getFloat ndarray i))
  ([ndarray i j] (.getFloat ndarray i j)))

(defn get-row
  "Get a copy of a row."
  [ndarray r] (.getRow ndarray r))

(defn get-scalar
  "Returns the element(s) at the specified index(indices)"
  ([ndarray i] (.getScalar ndarray i))
  ([ndarray i j] (.getScalar ndarray i j)))

(defn gt
  "Returns the binary ndarray for greater comparison."
  [ndarray ndarray2] (.gt ndarray ndarray2))

(defn gt!
  "Returns the binary ndarray for greater comparison, changes in place."
  [ndarray ndarray2] (.gti ndarray ndarray2))

(defn index
  "Return the linear index of the specified row and column."
  [ndarray row column] (.index ndarray row column))

(defn inner-most-stride
  "Get the inner most stride wrt the ordering of the array."
  [ndarray] (.innerMostStride ndarray))

(defn cleaned-up?
  "Returns true if the ndarray has already been freed."
  [ndarray] (.isCleanedUp ndarray))

(defn column-vector?
  "Checks whether the matrix is a column vector."
  [ndarray] (.isColumnVector ndarray))

(defn compressed?
  "Returns true if this array is compressed, and false otherwise."
  [ndarray] (.isCompressed ndarray))

(defn matrix?
  "Returns true if this ndarray is 2d or 3d with a singleton element."
  [ndarray] (.isMatrix ndarray))

(defn row-vector?
  "Checks whether the matrix is a row vector."
  [ndarray] (.isRowVector ndarray))

(defn scalar?
  "Test whether a matrix is scalar."
  [ndarray] (.isScalar ndarray))

(defn square?
  "Returns whether the matrix has the same rows and columns."
  [ndarray] (.isSquare ndarray))

(defn valid?
  "Returns whether the ndarray is valid or not."
  [ndarray] (.isValid ndarray))

(defn vector-array?
  "Checks whether the matrix is a vector."
  [ndarray] (.isVector ndarray))

(defn view?
  "Returns true if this array is a view or not."
  [ndarray] (.isView ndarray))

(defn wrap-around?
  "Returns true if the ndarray on linear indexing wraps around based on the stride(1) of the ndarray This is a useful optimization in linear view where strides that might otherwise go out of bounds but wrap around instead."
  [ndarray] (.isWrapAround ndarray))

(defn iterator
  "Returns a java iterator for the given array."
  [ndarray] (.iterator ndarray))

(defn length
  "Returns the total number of elements in the ndarray as an integer."
  [ndarray] (.length ndarray))

(defn length-long
  "Returns the total number of elements in the ndarray as a long integer."
  [ndarray] (.lengthLong ndarray))

(defn linear-index
  "Get the linear index of the data in to the array."
  [ndarray i] (.linearIndex ndarray i))

(defn linear-view-column-order
  "Returns a linear view reference of shape 1,length(ndarray)."
  [ndarray] (.linearViewColumnOrder ndarray))

(defn lt
  "Returns the binary ndarray for less comparison."
  [ndarray ndarray2] (.lt ndarray ndarray2))

(defn lt!
  "Returns the binary ndarray for less comparison, changes first array in place."
  [ndarray ndarray2] (.lti ndarray ndarray2))

(defn lte
  "Returns the binary ndarray for less than or equals comparison."
  [ndarray ndarray2] (.lte ndarray ndarray2))

(defn lte! 
  "Returns the binary ndarray for less than or equals comparison, changes first array in place."
  [ndarray ndarray2] (.ltei ndarray ndarray2))

(defn major-stride
  "Return the major stride for an ndarray."
  [ndarray] (.majorStride ndarray))

(defn max-num
  "Get the maximum number in the array."
  [ndarray] (.maxNumber ndarray))

(defn mean-num
  "Get the mean number of the array."
  [ndarray] (.meanNumber ndarray))

(defn min-num
  "Get the minimum number of the array."
  [ndarray] (.minNumber ndarray))

(defn neq
  "Returns the binary ndarray for not equals comparison."
  [ndarray ndarray2] (.neq ndarray ndarray2))

(defn neq!
  "Returns the binary ndarray for not equals comparison, changes first array in place."
  [ndarray ndarray2] (.neqi ndarray ndarray2))

(defn norm1-num
  "Returns the norm1 number"
  [ndarray] (.norm1Number ndarray))

(defn norm2-num
  "Returns the norm2 number"
  [ndarray] (.norm2Number ndarray))

(defn normmax-num
  "Returns the normmax number."
  [ndarray] (.normmaxNumber ndarray))

(defn offset
  "Returns the start of where the ndarray is for the underlying data"
  [ndarray] (.offset ndarray))

(defn ordering
  "Return the ordering (fortran or c) of this ndarray"
  [ndarray] (.ordering ndarray))

(defn original-offset
  "Returns the start of where the ndarray is for the original data buffer"
  [ndarray] (.originalOffset ndarray))

(defn put
  "Put the elements of the ndarray in to the specified indices."
  ([ndarray indices element] (.put ndarray indices element))
  ([ndarray i j element] (.put ndarray i j element)))

(defn put-column
  "Insert a column in to this array Will throw an exception if this ndarray is not a matrix"
  [ndarray column toput] (.putColumn ndarray column toput))

(defn put-row
  "Insert a row in to this array Will throw an exception if this ndarray is not a matrix"
  [ndarray row toput] (.putRow ndarray row toput))

(defn put-scalar
  "Insert a scalar float at the specified index (indices)"
  ([ndarray i value] (.putScalar ndarray i value))
  ([ndarray row col value] (.putScalar ndarray row col value))
  ([ndarray dim0 dim1 dim2 value] (.putScalar ndarray dim0 dim1 dim2 value))
  ([ndarray dim0 dim1 dim2 dim3 value] (.putScalar ndarray dim0 dim1 dim2 dim3 value)))

(defn put-slice
  "Assigns the given matrix (put) to the specified slice."
  [ndarray slice put] (.putSlice ndarray slice put))

(defn rank
  "Returns the rank of the ndarray (the number of dimensions)."
  [ndarray] (.rank ndarray))

(defn ravel
  "Flattens the array for linear indexing."
  ([ndarray] (.ravel ndarray))
  ([ndarray order] (.ravel ndarray order)))

(defn rdiv
  "Reverse division, returns new array."
  [ndarray ndarray2] (.rdiv ndarray ndarray2))

(defn rdiv!
  "Reverse division change first array in place."
  [ndarray ndarray2] (.rdivi ndarray ndarray2))

(defn rdiv-column-vector
  "Reverse divison of a column vector"
  [ndarray column-vector] (.rdivColumnVector ndarray column-vector))

(defn rdiv-column-vector!
  "Reverse divison of a column vector, changes in place."
  [ndarray column-vector] (.rdiviColumnVector ndarray column-vector))

(defn rdiv-row-vector
  "Reverse divison of a row vector"
  [ndarray row-vector] (.rdivRowVector ndarray row-vector))

(defn rdiv-row-vector!
  "Reverse divison of a row vector, changes in place."
  [ndarray row-vector] (.rdiviRowVector ndarray row-vector))

(defn remainder
  "Remainder operator"
  [ndarray denominator] (.remainder ndarray denominator))

(defn remainder! 
  "Remainder operator, changes in place."
  [ndarray denominator] (.remainderi ndarray denominator))

(defn repmat
  "Replicate and tile array to fill out to the given shape See: https://github.com/numpy/numpy/blob/master/numpy/matlib.py#L310-L358"
  [shape] (.repmat shape))

(defn reset-linear-view!
  "Resets the linear view"
  [ndarray] (.resetLinearView ndarray))

(defn rows
  "Returns the number of rows in the array (only 2d) throws an exception when called when not 2d"
  [ndarray] (.rows ndarray))

(defn rsub
  "Reverse subtraction"
  [ndarray ndarray2] (.rsub ndarray ndarray2))

(defn rsub!
  "Reverse subtraction, changes first array in place."
  [ndarray ndarray2] (.rsubi ndarray ndarray2))

(defn rsub-column-vector
  "Reverse subtraction of a column vector"
  [ndarray column-vector] (.rsubColumnVector ndarray column-vector))

(defn rsub-column-vector!
  "Reverse subtraction of a column vector, changes in place."
  [ndarray column-vector] (.rsubiColumnVector ndarray column-vector))

(defn rsub-row-vector
  "Reverse subtraction of a row vector."
  [ndarray row-vector] (.rsubRowVector ndarray row-vector))

(defn rsub-row-vector!
  "Reverse subtraction of a row vector, changes in place."
  [ndarray row-vector] (.rsubiRowVector ndarray row-vector))

(defn set-order!
  "Set the ordering."
  [ndarray order] (.setOrder ndarray order))

(defn set-stride!
  "Stride setter"
  [ndarray stride] (.setStride ndarray stride))

(defn size
  "Returns the size of this array along a particular dimension."
  [ndarray dimension] (.size ndarray dimension))

(defn slice
  "Returns the specified slice of this matrix, optionally along a dimension."
  ([ndarray s] (.slice ndarray s))
  ([ndarray s dimension] (.slice ndarray s dimension)))

(defn slices
  "Number of slices: aka shape[0]"
  [ndarray] (.slices ndarray))

(defn squared-distance
  "Returns the square of the Euclidean distance."
  [ndarray ndarray2] (.squaredDistance ndarray ndarray2))

(defn std-number
  "Standard deviation of an ndarray"
  [ndarray] (.stdNumber ndarray))

(defn mul
  "Multiplication of two arrays or scalar"
  [ndarray ndarray2] (.mul ndarray ndarray2))

(defn mul!
   "Multiplication of two arrays or scalar, changes in place."
  [ndarray ndarray2] (.muli ndarray ndarray2))

(defn stride
  "Returns the stride(indices along the linear index for which each slice is accessed) of this array"
  ([ndarray] (.stride ndarray))
  ([ndarray dimension] (.stride ndarray dimension)))

(defn sub
  "Subtraction of two matrices or scalar."
  [ndarray ndarray2] (.sub ndarray ndarray2))

(defn sub!
   "Subtraction of two matrices or scalar, changes first array in place."
  [ndarray ndarray2] (.subi ndarray ndarray2))

(defn sub-column-vector
  "Subtraction of a column vector"
  [ndarray column-vector] (.subColumnVector ndarray column-vector))

(defn sub-column-vector!
  "Subtraction of a column vector, changes in place."
  [ndarray column-vector] (.subiColumnVector ndarray column-vector))

(defn sub-row-vector
  "Subtraction of a row vector"
  [ndarray row-vector] (.subRowVector ndarray row-vector))

(defn sub-row-vector!
  "Subtraction of a row vector, changes in place."
  [ndarray row-vector] (.subiRowVector ndarray row-vector))

(defn swap-axes
  "Mainly here for people coming from numpy."
  [ndarray dimension with] (.swapAxes ndarray dimension with))

(defn to-string
  "Generate string representation of the matrix."
  [ndarray] (.toString ndarray))

(defn vector-along-dimension
  "Get the vector along a particular dimension"
  [ndarray index dimension]
  (.vectorAlongDimension ndarray index dimension))

(defn vectors-along-dimension
  "Returns the number of possible vectors for a given dimension"
  [ndarray dimension]
  (.vectorsAlongDimension ndarray dimension))


(defn vstack-arrays
  "Stacks a collection of ndarrays vertically (by row)
  Doesn't work if >2 X N, only 1 X N"
  [ndarrays]
  (let [shape (.shape (first ndarrays))
        new-array (Nd4j/create (count ndarrays) (second shape))]
    (doseq [n (range 0 (count ndarrays))]
      (.putRow new-array n (nth ndarrays n)))
    new-array))


(defn hstack-arrays
  "Stacks a collection of ndarrays horizontally"
  [ndarrays]
  (let [shape (.shape (first ndarrays))
        new-array (Nd4j/create (first shape) (count ndarrays))]
    (doseq [n (range 0 (count ndarrays))]
      (.putColumn new-array n (nth ndarrays n)))
    new-array))

;;Algorithms and other built formulas
(defn mean
  "Return mean vector"
  [ndarray]
  (let [shape (.shape ndarray)
        nrows (first shape)
        ncols (second shape)
        new-array (Nd4j/zeros 1 ncols)]
    (doseq [i (range nrows)]
      (.addi new-array (.getRow ndarray i)))
    new-array))

(defn covariance
  "Calculate covariance of ndarray."
  [ndarray]
    (let [average (mean ndarray)
          shape (.shape ndarray)
          sum (Nd4j/zeros (second shape) (second shape))]
      (doseq [i (range 0 (first shape))]
        (let [variance (.sub (.getRow ndarray i) average)
              row-covar (.mmul (.transpose (.dup variance)) variance)]
          (.addi sum row-covar)))         
      (.div sum (first shape))))

(defn svd-decomp
  "Single value decomposition of array, returns a map with keys :singular-values and :eigenvectors-transposed"
  [ndarray]
  (let [shape (.shape ndarray)
        rows (first shape)
        cols (second shape)
        s (Nd4j/create (if (< rows cols) rows cols))
        vt (Nd4j/create cols cols)]
    (.sgesvd (.lapack (Nd4j/getBlasWrapper))
      ndarray s nil vt)
    {:singular-values s :eigenvectors-transposed vt}))

(defn pca
  "Returns array compressed by principal component analysis to specified number of dimensions."
  [num-dims ndarray]
  (let [covar (covariance ndarray)
        svd-comps (svd-decomp covar)
        factors (->> (map-indexed (fn [i n] [n i]) (:singular-values svd-comps))
                     (sort-by first)
                     reverse
                     (take num-dims)
                     (map (fn [[eigenvalue id]] (.getColumn (:eigenvectors-transposed svd-comps) id)))
                     hstack-arrays)]
    (.mmul ndarray factors)))

(defn mean-vector
  [ndarray]
  (Nd4j/mean ndarray 0))

(defn normalize-zero!
  "Normalize array "
  [ndarray]
  (let [mn (Nd4j/mean ndarray 0)]
    (.subiRowVector ndarray mn)
    ndarray))

(defn max-index
  "Returns a vector like [max-number max-index]" 
  [ndarray]
  (first (sort-by second (map-indexed (fn [i n] [n i]) ndarray))))

(defn min-index
  "Returns a vector like [min-number min-index]"
  [ndarray]
  (last (sort-by second (map-indexed (fn [i n] [n i]) ndarray))))

;;Should this return a value or an array?
(defn inner-product
  "Takes two arrays and then returns a value as the inner product. Smaller shape."
  [a1 a2]
  (mmul a1 (transpose a2)))

(defn outer-product
  "Takes two arrays and then returns an array with a larger shape."
  [a1 a2]
  (mmul (transpose a1) a2))
