; benchmark generated from python API
(set-info :status unknown)
(declare-fun prefix_length () Int)
(declare-fun arbitrary_prefix_cell () Int)
(declare-fun isolation_key_0 () (Array Int Int))
(declare-fun future_key_0 () (Array Int Int))
(assert
 (> prefix_length 0))
(assert
 (< prefix_length 1024))
(assert
 (>= arbitrary_prefix_cell 0))
(assert
 (< arbitrary_prefix_cell prefix_length))
(assert
 (let ((?x457 (select isolation_key_0 arbitrary_prefix_cell)))
 (and (distinct ?x457 0) true)))
(assert
 (let ((?x448 (select future_key_0 arbitrary_prefix_cell)))
(= ?x448 0)))
(check-sat)
