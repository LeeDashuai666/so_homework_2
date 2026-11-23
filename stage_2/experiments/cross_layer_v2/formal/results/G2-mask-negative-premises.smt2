; benchmark generated from python API
(set-info :status unknown)
(declare-fun live_cursor () Int)
(declare-fun live_delta () Int)
(declare-fun live_token () Int)
(declare-fun observed_column () Int)
(assert
 (> live_cursor 0))
(assert
 (< live_cursor 1024))
(assert
 (>= live_delta (- 1024)))
(assert
 (<= live_delta 1024))
(assert
 (>= live_token 0))
(assert
 (< live_token 151936))
(assert
 (>= observed_column 0))
(assert
 (< observed_column 1024))
(check-sat)
