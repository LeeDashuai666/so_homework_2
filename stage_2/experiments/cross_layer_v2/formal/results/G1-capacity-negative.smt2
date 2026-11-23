; benchmark generated from python API
(set-info :status unknown)
(declare-fun prompt_length () Int)
(declare-fun decode_step () Int)
(assert
 (> prompt_length 0))
(assert
 (>= decode_step 0))
(assert
 (< decode_step 63))
(assert
 (<= prompt_length 1024))
(assert
 (let ((?x925 (+ prompt_length decode_step)))
(let (($x928 (>= ?x925 1024)))
(or (< ?x925 0) $x928))))
(check-sat)
