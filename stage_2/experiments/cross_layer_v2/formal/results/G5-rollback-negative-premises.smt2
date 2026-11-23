; benchmark generated from python API
(set-info :status unknown)
(declare-fun live_cursor () Int)
(declare-fun rollback_column () Int)
(assert
 (> live_cursor 0))
(assert
 (<= (+ live_cursor 64) 1024))
(assert
 (>= rollback_column 0))
(assert
 (< rollback_column live_cursor))
(check-sat)
