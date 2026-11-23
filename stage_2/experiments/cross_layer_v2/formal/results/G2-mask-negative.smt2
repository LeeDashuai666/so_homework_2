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
(assert
 (let ((?x1050 (+ live_cursor 1)))
(let (($x1427 (< observed_column ?x1050)))
(let ((?x1416 (lambda ((where_index Int) )(let ((?x1415 (lambda ((mask_index Int) )(let ((?x1050 (+ live_cursor 1)))
(let ((?x918 (lambda ((column_index Int) )column_index)
))
(let ((?x1039 (select ?x918 mask_index)))
(<= ?x1039 ?x1050)))))
))
(ite (select ?x1415 where_index) 0 (- 338953138925153547590470800371487866880))))
))
(let (($x690 (= (select ?x1416 observed_column) 0)))
(and (distinct $x690 $x1427) true))))))
(check-sat)
