(ns jutsu.matrix.core
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.ops.transforms Transforms]
           [org.nd4j.linalg.api.ndarray INDArray]))

;;http://nd4j.org/doc/org/nd4j/linalg/ops/transforms/Transforms.html
;;http://nd4j.org/doc/org/nd4j/linalg/api/ndarray/BaseNDArray.html
;;http://nd4j.org/doc/org/nd4j/linalg/factory/Nd4j.html

;;ND4J static methods
(defn zeros [rows cols] (Nd4j/zeros rows cols))

(defn jutsu-rows [coll]
  (if (instance? INDArray coll)
    (first (.shape coll))
    (if (coll? (first coll)) (count coll) 1)))

(defn jutsu-cols [coll]
  (if (instance? INDArray coll)
    (second (.shape coll))
    (if (coll? (first coll)) (count (first coll)) (count coll))))

(defn matrix
  "Converts clojure data to a ND4J array. Currently only supports 1 and 2 dimensional arrays."
  [coll]
  (let [h (jutsu-rows coll) w (jutsu-cols coll)]
    (if (= h 1)
      (Nd4j/create (float-array (seq coll)))
      (let [new-array (Nd4j/create h w)]
        (doseq [i (range 0 h)]
          (.putRow new-array i (Nd4j/create (float-array (nth coll i)))))
        new-array))))

(defn diag
  ([ndarray] (Nd4j/diag ndarray))
  ([ndarray k] (Nd4j/diag ndarray k)))

(defn from-byte-array [byte-array] (Nd4j/fromByteArray byte-array))

(defn clear-nans [ndarray] (Nd4j/clearNans ndarray))

(defn get-backend [] (Nd4j/getBackend))

(defn get-compressor [] (Nd4j/getCompressor))

(defn get-fft [] (Nd4j/getFFt))

(defn fallback-mode-enabled? [] (Nd4j/isFallbackModeEnabled))

(defn ones [rows cols] (Nd4j/ones rows cols))

(defn prod 
  ([ndarray] (Nd4j/prod ndarray))
  ([ndarray k] (Nd4j/prod ndarray k)))

(defn rand 
  ([shape] (Nd4j/rand shape))
  ([rows cols] (Nd4j/rand cols))
  ([rows cols seed] (Nd4j/rand cols))
  ([rows cols min max rng] (Nd4j/rand cols min max rng)))

(defn randn
  ([shape] (Nd4j/randn shape))
  ([rows columns] (Nd4j/randn rows columns))
  ([rows columns rng] (Nd4j/randn rows columns rng)))

(defn repeat [ndarray num] (Nd4j/repeat ndarray num))

(defn roll-axis 
  ([ndarray axis] (Nd4j/rollAxis ndarray axis))
  ([ndarray axis start] (Nd4j/rollAxis ndarray axis start)))

(defn rot [ndarray] (Nd4j/rot ndarray))

(defn rot-90 [ndarray] (Nd4j/rot90 ndarray))

(defn scalar
  ([value] (Nd4j/scalar value))
  ([value offset] (Nd4j/scalar value offset)))

(defn sort [ndarray dimension ascending] (Nd4j/sort ndarray dimension ascending))

(defn sort-columns [ndarray row-idx ascending] (Nd4j/sortColumns ndarray row-idx ascending))

(defn sort-rows [ndarray col-idx ascending] (Nd4j/sortRows ndarray col-idx ascending))

(defn sort-with-indices [ndarray dimension ascending] (Nd4j/sortWithIndices ndarray dimension ascending))

(defn sum 
  ([ndarray] (Nd4j/sum ndarray))
  ([ndarray dimension] (Nd4j/sum ndarray dimension)))

(defn to-byte-array [ndarray] (Nd4j/toByteArray ndarray))

(defn to-flatten [ndarrays] (Nd4j/toFlattened ndarrays))

(defn value-array-of
  ([shape value] (Nd4j/valueArrayOf shape value))
  ([rows columns value] (Nd4j/valueArrayOf rows columns value)))

(defn var
  ([ndarray] (Nd4j/var ndarray))
  ([ndarray dimension] (Nd4j/var ndarray dimension)))

;;Transform static methods
(defn abs [ndarray] (Transforms/abs ndarray true))

(defn abs! [ndarray] (Transforms/abs ndarray false))

(defn sigmoid [ndarray] (Transforms/sigmoid ndarray true))

(defn sigmoid! [ndarray] (Transforms/sigmoid ndarray false))

(defn tanh [ndarray] (Transforms/tanh ndarray true))

(defn tanh! [ndarray] (Transforms/tanh ndarray false))

(defn sqrt [ndarray] (Transforms/sqrt ndarray true))

(defn sqrt! [ndarray] (Transforms/sqrt ndarray))

(defn exp [ndarray] (Transforms/exp ndarray true))

(defn exp! [ndarray] (Transforms/exp ndarray false))

(defn and [x y] (Transforms/and x y))

(defn acos [ndarray] (Transforms/acos ndarray true))

(defn acos! [ndarray] (Transforms/acos ndarray false))

(defn asin [ndarray] (Transforms/asin ndarray true))

(defn asin! [ndarray] (Transforms/asin ndarray false))

(defn atan [ndarray] (Transforms/atan ndarray true))

(defn atan! [ndarray] (Transforms/atan ndarray false))

(defn ceil [ndarray] (Transforms/ceil ndarray true))

(defn ceil! [ndarray] (Transforms/ceil ndarray false))

(defn ceiling [ndarray] (Transforms/ceiling ndarray true))

(defn ceiling! [ndarray] (Transforms/cos ndarray false))

(defn cos [ndarray] (Transforms/ceiling ndarray true))

(defn cos! [ndarray] (Transforms/cos ndarray false))

(defn cosine-sim [ndarray ndarray2] (Transforms/cosineSim ndarray ndarray2))

(defn eps [ndarray] (Transforms/eps ndarray true))

(defn eps! [ndarray] (Transforms/eps ndarray false))

(defn floor [ndarray] (Transforms/floor ndarray true))

(defn floor! [ndarray] (Transforms/floor ndarray false))

(defn greater-than-or-equal [ndarray ndarray2] (Transforms/greaterThanOrEqual ndarray ndarray2 true))

(defn greater-than-or-equal! [ndarray ndarray2] (Transforms/greaterThanOrEqual ndarray ndarray2 false))

(defn hard-tanh [ndarray] (Transforms/hardTanh ndarray true))

(defn hard-tanh! [ndarray] (Transforms/hardTanh ndarray false))

(defn identity [ndarray] (Transforms/identity ndarray true))

(defn identity! [ndarray] (Transforms/identity ndarray false))

(defn leakyRelu [ndarray] (Transforms/leakyRelu ndarray true))

(defn leakyRelu! [ndarray] (Transforms/leakyRelu ndarray false))

(defn less-than-or-equal [ndarray ndarray2] (Transforms/lessThanOrEqual ndarray ndarray2 true))

(defn less-than-or-equal! [ndarray ndarray2] (Transforms/lessThanOrEqual ndarray ndarray2 false))

(defn log 
  ([ndarray] (Transforms/log ndarray true))
  ([ndarray base] (Transforms/log ndarray base true)))

(defn log! 
  ([ndarray] (Transforms/log ndarray false))
  ([ndarray base] (Transforms/log ndarray base false)))

(defn manhattan-distance [ndarray ndarray2]
  (Transforms/manhattanDistance ndarray ndarray2))

(defn max
  "k can also be second array"
  ([ndarray k] (Transforms/max ndarray k true)))

(defn max! [ndarray k] (Transforms/max ndarray k false))

(defn min [ndarray k] (Transforms/min ndarray k true))

(defn min! [ndarray k] (Transforms/min ndarray k false))

(defn neg [ndarray] (Transforms/neg ndarray true))

(defn neg! [ndarray] (Transforms/neg ndarray false))

(defn normalize! [ndarray] (Transforms/normalizeZeroMeanAndUnitVariance ndarray))

(defn not [ndarray] (Transforms/not ndarray))

(defn or [ndarray ndarray2] (Transforms/or ndarray ndarray2))

(defn pow [ndarray power] (Transforms/pow ndarray power true))

(defn pow! [ndarray power] (Transforms/pow ndarray power false))

(defn relu [ndarray] (Transforms/relu ndarray true))

(defn relu! [ndarray] (Transforms/relu ndarray false))

(defn round [ndarray] (Transforms/round ndarray true))

(defn round! [ndarray] (Transforms/round ndarray false))

(defn sign [ndarray] (Transforms/sign ndarray true))

(defn sign! [ndarray] (Transforms/sign ndarray false))

(defn sin [ndarray] (Transforms/sin ndarray true))

(defn sin! [ndarray] (Transforms/sin ndarray false))

(defn soft-plus [ndarray] (Transforms/softPlus ndarray true))

(defn soft-plus! [ndarray] (Transforms/softPlus ndarray false))

(defn stabilize [ndarray k] (Transforms/stabilize ndarray k true))

(defn stabilize! [ndarray k] (Transforms/stabilize ndarray k false))

(defn unit-vec [ndarray] (Transforms/unitVec ndarray))

(defn xor [ndarray ndarray2] (Transforms/xor ndarray ndarray2))

;;ND4j Methods
(defn linear-view [ndarray] (.linearView ndarray))

(defn transpose [ndarray] (.transpose ndarray))

(defn mmul [ndarray ndarray2] (.mmul ndarray ndarray2))

(defn mmul! [ndarray ndarray2] (.mmuli ndarray ndarray2))

(defn mul-column-vector [ndarray column-vector] (.mulColumnVector ndarray column-vector))

(defn mul-column-vector! [ndarray column-vector] (.muliColumnVector ndarray column-vector))

(defn mul-row-vector [ndarray row-vector] (.mulRowVector ndarray row-vector))

(defn mul-row-vector! [ndarray row-vector] (.muliRowVector ndarray row-vector))

(defn reshape [ndarray rows cols] (.reshape ndarray rows cols))

(defn add [ndarray ndarray2] (.add ndarray ndarray2))

(defn add! [ndarray ndarray2] (.addi ndarray ndarray2))

(defn add-column-vector [ndarray column-vector]
  (.addColumnVector ndarray column-vector))

(defn add-column-vector! [ndarray column-vector]
  (.addiColumnVector ndarray column-vector))

(defn add-row-vector [ndarray row-vector]
  (.addRowVector ndarray row-vector))

(defn add-row-vector! [ndarray row-vector]
  (.addiRowVector ndarray row-vector))

(defn assign [ndarray] (.assign ndarray))

(defn check-dimensions [ndarray ndarray2] (.ndarray ndarray2))

(defn cleanup [ndarray] (.cleanup ndarray))

(defn columns [ndarray] (.columns ndarray))

(defn cumsum [ndarray dimension] (.cumsum ndarray dimension))

(defn cumsum! [ndarray dimension] (.cumsumi ndarray dimension))

(defn get-data [ndarray] (.data ndarray))

(defn one-norm-distance [ndarray ndarray2] (.distance1 ndarray ndarray2))

(defn div [ndarray ndarray2] (.div ndarray ndarray2))

(defn div! [ndarray ndarray2] (.divi ndarray ndarray2))

(defn div-column-vector [ndarray column-vector] (.divColumnVector ndarray column-vector))

(defn div-column-vector! [ndarray column-vector] (.diviColumnVector ndarray column-vector))

(defn div-row-vector [ndarray row-vector] (.divRowVector ndarray row-vector))

(defn div-row-vector! [ndarray row-vector] (.diviRowVector ndarray row-vector))

(defn dup
  ([ndarray] (.dup ndarray))
  ([ndarray order] (.dup ndarray order)))

(defn element-stride [ndarray] (.elementStride ndarray))

(defn element-wise-stride [ndarray] (.element-wise-stride ndarray))

(defn eq [ndarray ndarray2] (.eq ndarray ndarray2))

(defn eq! [ndarray ndarray2] (.eqi ndarray ndarray2))

(defn equals? [ndarray ndarray2] (.equals ndarray ndarray2))

(defn equals-with-eps? [ndarray ndarray2 eps-value] (.equalsWithEps ndarray ndarray2 eps-value))

(defn fmod [ndarray ndarray2] (.fmod ndarray ndarray2))

(defn fmod! [ndarray ndarray2] (.fmodi ndarray ndarray2))

(defn get-column [ndarray c] (.getColumn ndarray c))

(defn get-double
  ([ndarray i] (.getDouble ndarray i))
  ([ndarray i j] (.getDouble ndarray i j)))

(defn get-float
  ([ndarray i] (.getFloat ndarray i))
  ([ndarray i j] (.getFloat ndarray i j)))

(defn get-row [ndarray r] (.getRow ndarray r))

(defn get-scalar
  ([ndarray i] (.getScalar ndarray i))
  ([ndarray i j] (.getScalar ndarray i j)))

(defn gt [ndarray ndarray2] (.gt ndarray ndarray2))

(defn gt! [ndarray ndarray2] (.gti ndarray ndarray2))

(defn index [ndarray row column] (.index ndarray row column))

(defn inner-most-stride [ndarray] (.innerMostStride ndarray))

(defn cleaned-up? [ndarray] (.isCleanedUp ndarray))

(defn column-vector? [ndarray] (.isColumnVector ndarray))

(defn compressed? [ndarray] (.isCompressed ndarray))

(defn matrix? [ndarray] (.isMatrix ndarray))

(defn row-vector? [ndarray] (.isRowVector ndarray))

(defn scalar? [ndarray] (.isScalar ndarray))

(defn square? [ndarray] (.isSquare ndarray))

(defn valid? [ndarray] (.isValid ndarray))

(defn vector? [ndarray] (.isVector ndarray))

(defn view? [ndarray] (.isView ndarray))

(defn wrap-around? [ndarray] (.isWrapAround ndarray))

(defn iterator [ndarray] (.iterator ndarray))

(defn length [ndarray] (.length ndarray))

(defn length-long [ndarray] (.lengthLong ndarray))

(defn linear-index [ndarray i] (.linearIndex ndarray i))

(defn linear-view-column-order [ndarray] (.linearViewColumnOrder ndarray))

(defn lt [ndarray ndarray2] (.lt ndarray ndarray2))

(defn lt! [ndarray ndarray2] (.lti ndarray ndarray2))

(defn lte [ndarray ndarray2] (.lte ndarray ndarray2))

(defn lte! [ndarray ndarray2] (.ltei ndarray ndarray2))

(defn major-stride [ndarray] (.majorStride ndarray))

(defn max-number [ndarray] (.maxNumber ndarray))

(defn mean-number [ndarray] (.meanNumber ndarray))

(defn min-number [ndarray] (.minNumber ndarray))

(defn neq [ndarray ndarray2] (.neq ndarray ndarray2))

(defn neq! [ndarray ndarray2] (.neqi ndarray ndarray2))

(defn norm1-number [ndarray] (.norm1Number ndarray))

(defn norm2-number [ndarray] (.norm2Number ndarray))

(defn normmax-number [ndarray] (.normmaxNumber ndarray))

(defn offset [ndarray] (.offset ndarray))

(defn ordering [ndarray] (.ordering ndarray))

(defn original-offset [ndarray] (.originalOffset ndarray))

(defn put
  ([ndarray indices element] (.put ndarray indices element))
  ([ndarray i j element] (.put ndarray i j element)))

(defn put-column [ndarray column toput] (.putColumn ndarray column toput))

(defn put-row [ndarray row toput] (.putRow ndarray row toput))

(defn put-scalar
  ([ndarray i value] (.putScalar ndarray i value))
  ([ndarray row col value] (.putScalar ndarray row col value))
  ([ndarray dim0 dim1 dim2 value] (.putScalar ndarray dim0 dim1 dim2 value))
  ([ndarray dim0 dim1 dim2 dim3 value] (.putScalar ndarray dim0 dim1 dim2 dim3 value)))

(defn put-slice [ndarray slice put] (.putSlice ndarray slice put))

(defn rank [ndarray] (.rank ndarray))

(defn ravel 
  ([ndarray] (.ravel ndarray))
  ([ndarray order] (.ravel ndarray order)))

(defn rdiv [ndarray ndarray2] (.rdiv ndarray ndarray2))

(defn rdiv! [ndarray ndarray2] (.rdivi ndarray ndarray2))

(defn rdiv-column-vector [ndarray column-vector] (.rdivColumnVector ndarray column-vector))

(defn rdiv-column-vector! [ndarray column-vector] (.rdiviColumnVector ndarray column-vector))

(defn rdiv-row-vector [ndarray row-vector] (.rdivRowVector ndarray row-vector))

(defn rdiv-row-vector! [ndarray row-vector] (.rdiviRowVector ndarray row-vector))

(defn remainder [ndarray denominator] (.remainder ndarray denominator))

(defn remainder! [ndarray denominator] (.remainderi ndarray denominator))

(defn repmat [shape] (.repmat shape))

(defn reset-linear-view! [ndarray] (.resetLinearView ndarray))

(defn rows [ndarray] (.rows ndarray))

(defn rsub [ndarray ndarray2] (.rsub ndarray ndarray2))

(defn rsub! [ndarray ndarray2] (.rsubi ndarray ndarray2))

(defn rsub-column-vector [ndarray column-vector] (.rsubColumnVector ndarray column-vector))

(defn rsub-column-vector! [ndarray column-vector] (.rsubiColumnVector ndarray column-vector))

(defn rsub-row-vector! [ndarray row-vector] (.rsubiRowVector ndarray row-vector))

(defn set-order! [ndarray order] (.setOrder ndarray order))

(defn set-stride! [ndarray stride] (.setStride ndarray stride))

(defn size [ndarray dimension] (.size ndarray dimension))

(defn slice
  ([ndarray s] (.slice ndarray s))
  ([ndarray s dimension] (.slice ndarray s dimension)))

(defn slices [ndarray] (.slice ndarray))

(defn squared-distance [ndarray ndarray2] (.squaredDistance ndarray ndarray2))

(defn std-number [ndarray] (.stdNumber ndarray))

(defn stride
  ([ndarray] (.stride ndarray))
  ([ndarray dimension] (.stride ndarray dimension)))

(defn sub [ndarray ndarray2] (.sub ndarray ndarray2))

(defn sub! [ndarray ndarray2] (.subi ndarray ndarray2))

(defn sub-column-vector [ndarray column-vector] (.subColumnVector ndarray column-vector))

(defn sub-column-vector! [ndarray column-vector] (.subiColumnVector ndarray column-vector))

(defn sub-row-vector [ndarray row-vector] (.subRowVector ndarray row-vector))

(defn sub-row-vector! [ndarray row-vector] (.subiRowVector ndarray row-vector))

(defn sum-number [ndarray] (.sumNumber ndarray))

(defn swap-axes [ndarray dimension with] (.swapAxes ndarray dimension with))

(defn to-string [ndarray] (.toString ndarray))

(defn transpose! [ndarray] (.transposei ndarray))

(defn vector-along-dimension [ndarray index dimension]
  (.vectorAlongDimension ndarray index dimension))

(defn vectors-along-dimension [ndarray dimension]
  (.vectorsAlongDimension ndarray dimension))


;;Algorithms and other built formulas

(defn mean [ndarray]
  (let [shape (.shape ndarray)
        nrows (first shape)
        ncols (second shape)
        new-array (Nd4j/zeros 1 ncols)]
    (doseq [i (range nrows)]
      (.addi new-array (.getRow ndarray i)))
    new-array))

(defn covariance [ndarray]
    (let [average (mean ndarray)
          shape (.shape ndarray)
          sum (Nd4j/zeros (second shape) (second shape))]
      (doseq [i (range 0 (first shape))]
        (let [variance (.sub (.getRow ndarray i) average)
              row-covar (.mmul (.transpose (.dup variance)) variance)]
          (.addi sum row-covar)))         
      (.div sum (first shape))))

(defn svd-decomp [ndarray]
  (let [shape (.shape ndarray)
        rows (first shape)
        cols (second shape)
        s (Nd4j/create (if (< rows cols) rows cols))
        vt (Nd4j/create cols cols)]
    (.sgesvd (.lapack (Nd4j/getBlasWrapper))
      ndarray s nil vt)
    {:singularvalues s :eigenvectors_transposed vt}))

(defn pca [num-dims ndarray]
  (let [covar (covariance ndarray)
        svd-comps (svd-decomp covar)
        factors (->> (map-indexed (fn [i n] [n i]) (:singularvalues svd-comps))
                     (sort-by first)
                     reverse
                     (take num-dims)
                     (map (fn [[eigenvalue id]] (.getColumn (:eigenvectors_transposed svd-comps) id)))
                     hstack-arrays)]
    (.mmul ndarray factors)))

(defn normalize-zero! [ndarray]
  (let [mn (Nd4j/mean ndarray 0)]
    (.subiRowVector ndarray mn)
    ndarray))
