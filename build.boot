(set-env!
  :resource-paths #{"src"}
  :dependencies '[[org.clojure/clojure "1.9.0"]
                  [nightlight "1.7.0" :scope "test"]
                  [adzerk/boot-test "1.2.0" :scope "test"]
                  [org.nd4j/nd4j-native-platform "1.0.0-beta2" :scope "test"]
                  [org.nd4j/nd4j-api "1.0.0-beta2"]
                  [boot-codox "0.10.3" :scope "test"]]
  :repositories (conj (get-env :repositories)
                      ["clojars" {:url "https://clojars.org/repo"
                                  :username (System/getenv "CLOJARS_USER")
                                  :password (System/getenv "CLOJARS_PASS")}]))

(task-options!
  jar {:main 'jutsu.matrix.core
       :manifest {"Description" "jutsu.matrix is a linear algebra library meant for the jutsu data science framework"}}
  pom {:version "0.0.15"
       :project 'hswick/jutsu.matrix
       :description "jutsu.matrix is a linear algebra library meant for the jutsu data science framework"
       :url "https://github.com/hswick/jutsu.matrix"}
  push {:repo "clojars"})

(deftask deploy []
  (comp
    (pom)
    (jar)
    (install)
    (push)))

;;So nightlight can still open even if there is an error in the core file
(try
  (require 'jutsu.matrix.core)
  (catch Exception e (.getMessage e)))

(require
  '[nightlight.boot :refer [nightlight]]
  '[adzerk.boot-test :refer :all]
  '[codox.boot :refer [codox]])

(deftask night []
  (comp
    (wait)
    (nightlight :port 4000)))

(deftask testing [] (merge-env! :resource-paths #{"test"}) identity)

(deftask test-code
  []
  (comp
    (testing)
    (test)))

(deftask gen-docs
  []
  (set-env! :source-paths #(conj % "docs"))
  (comp
    (codox :name "jutsu.matrix"
      :description "Clojure library for linear algebra operations, wraps ND4J."
      :version "0.0.15"
      :source-paths #{"src/jutsu/matrix/"}
      :output-path "docs")
    (target)))
