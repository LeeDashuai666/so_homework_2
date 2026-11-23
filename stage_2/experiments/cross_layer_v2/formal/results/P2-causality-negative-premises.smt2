; benchmark generated from python API
(set-info :status unknown)
(declare-fun boundary () Int)
(declare-fun current_position () Int)
(declare-fun suffix_dependency_previous_layer () (Array Int Bool))
(assert
 (> boundary 0))
(assert
 (>= current_position 0))
(assert
 (< current_position boundary))
(assert
 (forall ((dependency_position Int) )(let (($x40 (select suffix_dependency_previous_layer dependency_position)))
 (let (($x41 (not $x40)))
 (=> (and (>= dependency_position 0) (< dependency_position boundary)) $x41))))
 )
(assert
 (select suffix_dependency_previous_layer boundary))
(check-sat)
