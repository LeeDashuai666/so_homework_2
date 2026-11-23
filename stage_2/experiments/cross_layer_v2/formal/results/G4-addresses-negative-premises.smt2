; benchmark generated from python API
(set-info :status unknown)
(declare-fun prefix_length () Int)
(assert
 (> prefix_length 0))
(assert
 (< prefix_length 1024))
(check-sat)
