; benchmark generated from python API
(set-info :status unknown)
(declare-fun prefix_length () Int)
(declare-fun arbitrary_prefix_cell () Int)
(assert
 (> prefix_length 0))
(assert
 (< prefix_length 1024))
(assert
 (>= arbitrary_prefix_cell 0))
(assert
 (< arbitrary_prefix_cell prefix_length))
(check-sat)
