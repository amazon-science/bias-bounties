import numpy as np
import copy
import pandas as pd
# Data structure for the decision list


class DecisionlistNode:
    """

        Node of decision lists are objects with the following form

            -----------------        ------------------
            |   predicate   | --0--> | right_child Node|
            -----------------        ------------------
                    |
                    1
                    |
                    v
                  Leaf
    """

    def __init__(self, predicate=None, leaf=None, right_child=None):
        """
        Instantiates DecisionListNode on input settings

        Parameters
        ----------
        predicate: function from X to {0,1}. If not defined in input to initialization, will be defined as function that
            always returns 1
        leaf: classification function from X to {0,1}. If not defined in the initialization, will be defined as function
            that always returns 1.
        right_child: Node object
        """
        self.predicate = predicate
        self.leaf = leaf
        self.right_child = right_child

        if self.predicate is None:
            self.predicate = lambda x: np.ones(len(x))

        if self.leaf is None:
            self.leaf = lambda x: np.ones(len(x))


class DecisionList:
    """
    A simple decision list object with standard traversal

            -----------------        ------------------                  ------------------
            |   predicate 1  | --0--> |   predicate 2  | --0--> ... ---> |   predicate n  |  ---> 0
            -----------------        ------------------                  ------------------
                    |						  |                                  |
                    1                         1                                  1
                    |                         |                                  |
                    v                         v                                  v
                  Leaf 1                    Leaf 2                             Leaf n

    """

    def __init__(self, initial_model):
        self.head = DecisionlistNode(leaf=initial_model)
        self.curr_node = self.head
        self.predicates = [self.head.predicate]
        self.leaves = [self.head.leaf]

    def predict(self, X):
        """
        Predicts the label of x according to traversal of the decision list.
        """
        if self.curr_node.predicate(X) == 1:
            # set output to the leaf function evaluated at x
            out = self.curr_node.leaf(X)
            # reset current node to head
            self.curr_node = self.head
            return out

        elif self.curr_node.right_child is not None:
            self.curr_node = self.curr_node.right_child
            return self.predict(X)
        else:
            # reset current node to head and output 0 with a warning
            print("Warning: reached end of PDL with no predicate succeeding, returning 0.")
            return 0

    def prepend(self, node):
        """
        Prepends a new node to the head of the decision list.

        node we want to                          current
        prepend                               decision list
        ---------                              -------------      -----------------------
        | node | --> node.right_child = None    | self.head | ---> | self.head.right_child| --> ...
        --------                               -------------      -----------------------
            |						               |
            1                                      1
            |                                      |
            v                                      v
          leaf                                self.head.leaf
        """
        node.right_child = self.head
        self.predicates.append(node.predicate)
        self.leaves.append(node.leaf)
        self.head = node
        self.curr_node = self.head


class PointerDecisionListNode(DecisionlistNode):
    """

        Node of pointer decision lists are objects with the following form if catch_node = False

            -----------------        ------------------
            |   predicate   | --0--> | right_child Node|
            -----------------        ------------------
                    |
                    1
                    |
                    v
                  Leaf

        Or, if catch_node = True:

            -----------------        ------------------
            |   predicate   | --0--> | right_child Node|
            -----------------        ------------------
                    |
                    1
                    |
                    v
            -----------------
            | right_main_node |
            -----------------
    """

    def __init__(self, predicate=None, leaf=None, right_child=None, catch_node=False, right_main_node=None,pred_name=None, data=pd.DataFrame()):
        DecisionlistNode.__init__(self, predicate=predicate, leaf=leaf, right_child=right_child)

        self.catch_node = catch_node
        self.right_main_node = right_main_node
        self.pred_name = pred_name
        self.data = data
        if catch_node:
            self.leaf = None


class PointerDecisionList:
    """
                     -------------------------------------------------------------------------------
                    |                         |                         						   |
                    1						  1                         						   v
                      |                         |                  right_main_node		    right_main_node.right_child
            -----------------        ------------------         ------------------          ---------------           -----------------
            |   predicate 1  | --0--> |   predicate 2  | --0--> |   predicate 3  |  --0-->  | predicate 3 |    ...    |  predicate n  |  ---> 0
            -----------------        ------------------         ------------------          ---------------           ------------------
                catch_node    			 catch_node	                    |                          |                          |
                                                                        1                          1                          1
                                                                        |                          |                          |
                                                                        v                          v                          v
                                                                           Leaf                       Leaf                        Leaf

    """

    def __init__(self, initial_model, all_groups=None):
        """
        initial_model is some fit function e.g. a .fit method output by scikit.learn,
        which takes as input a dataframe x and outputs labels y for those values of x.

        If you already know the groups we'll be passing in, they can be specified with all_groups
        """
        self.all_groups = None
        if all_groups is None:
            all_groups = []
        self.head = PointerDecisionListNode(leaf=initial_model, pred_name='Total')
        self.curr_node = self.head
        self.predicates = [self.head.predicate]
        self.pred_names = ['Total']
        self.leaves = [self.head.leaf]

        # keeping track of the group errors so far here (including groups that haven't been introduced yet, which
        # wouldn't happen IRL!) for computational efficiency.
        # only relevant if we already know the groups we're computing things on.
        if all_groups:
            n = len(all_groups)
            self.test_errors = np.empty(shape=(n, n))
            self.test_errors[:] = np.NaN
            self.train_errors = np.empty(shape=(n, n))
            self.train_errors[:] = np.NaN
        else:
            self.test_errors = []
            self.train_errors = []

        # keeping track of the number of rounds so far
        self.num_rounds = 0

        # keeping track of the final node belonging to each update so that we can point to those
        self.update_nodes = [self.head]
        self.track_rejects = [1]
        self.update_node_indices_tracking_rejects = [0]

    def init_groups(self):
        n = len(self.all_groups)
        self.test_errors = np.empty(shape=(n, n))
        self.test_errors[:] = np.NaN  # filling with NaNs to avoid confusion
        self.train_errors = np.empty(shape=(n, n))
        self.train_errors[:] = np.NaN  # filling with NaNs to avoid confusion

    def predict(self, X):
        """
        Predicts the labels of X according to the traversal of the pointer decision list.
        The function starts with all data X on the head node, and splits the dataframe by
        group membership, sending the sliced data along the proper pointers and stores the
        sliced data in the node it was sent to. After all data has been sent from the head
        node, the function moves to the right child of the node and iterates this process
        until it reaches the end of the PDL. If a hypothesis is made, the predictions are
        stored in a side array maintaining their position with respect to the initial
        dataframe.
        """
        # initialize pointer to head node
        self.curr_node = self.head

        # set head node data to be the entire dataframe
        self.curr_node.data = X

        # create deepcopy in order to keep from creating updates on X
        Y = copy.deepcopy(X)

        # set flag predictions
        Y['predictions'] = 2

        # initialize loop on right child existence
        while self.curr_node.right_child is not None:
            # if catch node then group members get sent down the PDL
            if self.curr_node.catch_node is True:
                # if there is data on this node, send it to where it should go according to group membership
                if self.curr_node.data.empty == False:
                    # if in group, send down the PDL to the proper location
                    if (self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1]).empty == False:
                        # add data from group to this node for later
                        self.curr_node.right_main_node.data = self.curr_node.right_main_node.data.append(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1])
                    # if not in group, send to node on the right of current node
                    if (self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 0]).empty == False:
                        # add data to right node dataset
                        self.curr_node.right_child.data = self.curr_node.right_child.data.append(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 0])
                # reset current node data to empty since all data was sent along paths
                self.curr_node.data = pd.DataFrame()

                # walk to the right child of the current node and iterate
                self.curr_node = self.curr_node.right_child
            # otherwise, traverse without pointers
            else:
                if self.curr_node.data.empty == False:
                    # if data on the current node is not in the group, send to the right child
                    if (self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 0]).empty == False:
                        self.curr_node.right_child.data = self.curr_node.right_child.data.append(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 0])
                    # if data on the current node is in the group, predict on it
                    if (self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1]).empty == False:
                        # update predictions list based on location of sliced data in the original dataframe
                        Y['predictions'].loc[self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1].index] = self.curr_node.leaf(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1])
                # reset current node data to empty
                self.curr_node.data = pd.DataFrame()

                # walk to the right child of the current node and iterate
                self.curr_node = self.curr_node.right_child

        # if there is no right child, we are on the final node and should run once more time
        if self.curr_node.data.empty == False:
            # update predictions
            Y['predictions'].loc[self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1].index] = self.curr_node.leaf(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 1])
            self.curr_node.data = self.curr_node.data.loc[(self.curr_node.data[self.curr_node.predicate(self.curr_node.data) == 0]).index]

            # test case: if there is data left in the current node, this means that some data did not get predicted upon during PDL traversal
            if (not self.curr_node.data.empty):
                print(
                    "Warning: reached end of PDL with no predicate succeeding, returning 0.")
                self.curr_node.data = pd.DataFrame()
                return 0

        # test case: if there is a prediction which still has value 2 in the predictions list, it never was updated and something went wrong
        if (Y['predictions'] == 2).sum() > 0:
            # reset current node to head and output 0 with a warning
            print("Warning: reached end of PDL with no predicate succeeding, returning 0.")
            self.curr_node.data = pd.DataFrame()
            return 0
        else:
            # output predictions
            predictions = Y['predictions']
            self.curr_node.data = pd.DataFrame()
            return predictions

    def prepend(self, node):
        """
        Prepends a new node to the head of the decision list.
        """
        if node.catch_node is True:
            # if prepending to another catch_node
            if self.head.catch_node is True:
                node.right_child = self.head
                self.head = node
                self.curr_node = self.head
            else:
                node.right_child = self.head
                self.head = node
                self.curr_node = self.head
        else:
            node.right_child = self.head
            self.predicates.append(node.predicate)
            self.pred_names.append(node.pred_name)
            self.leaves.append(node.leaf)
            self.head = node
            self.curr_node = self.head

    def pop(self):
        """
        Removes the head node from the decision list
        """
        self.head = self.head.right_child
        self.curr_node = self.head
