; benchmark generated from python API
(set-info :status unknown)
(declare-fun model_b () Int)
(declare-fun model_a () Int)
(declare-fun pixel_b () Int)
(declare-fun pixel_a () Int)
(declare-fun grid_b () Int)
(declare-fun grid_a () Int)
(declare-fun prefix_tokens_b () Int)
(declare-fun prefix_tokens_a () Int)
(declare-fun prefix_positions_b () Int)
(declare-fun prefix_positions_a () Int)
(assert
 (= model_a model_b))
(assert
 (= pixel_a pixel_b))
(assert
 (= grid_a grid_b))
(assert
 (= prefix_tokens_a prefix_tokens_b))
(assert
 (and (distinct prefix_positions_a prefix_positions_b) true))
(check-sat)
