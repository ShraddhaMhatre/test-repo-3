Tree * add(Tree* nod, int number) {
    if (nod == NULL) {
        nod = (Tree*) malloc(sizeof (Tree));
        if (nod == NULL) {
            return NULL;
        }
        nod->info.num = number;
        nod->left = NULL;
        nod->right = NULL;
        return nod;
    }
    if (nod->info.num > number) {
        nod->left = add(nod->left, number);
    } else {
        nod->right = add(nod->right, number);
    }
    return nod;
}

void printInorder(Tree* nod) {
    if (nod == NULL) {
        return;
    }
    printInorder(nod->left);
    printf(" %d ", nod->info.num);
    printInorder(nod->right);
}

Tree * findMinimum(Tree *t) {
    if (t == NULL) {
        return t;
    }
    if (t->left == NULL) {
        return t;
    }
    return findMinimum(t->left);
}

Tree * delete(Tree *t, Item x) {
    Tree * tmp;
    if (t == NULL) {
        return t;
    }
    if (x.num < t->info.num) {
        t->left = delete(t->left, x);
    } else {
        if (x.num > t->info.num) {
            t->right = delete(t->right, x);
        } else {
            if (t->left && t->right) {
                tmp = findMinimum(t->right);
                t->info.num = tmp->info.num;
                t->right = delete(t->right, t->info);
            } else {
                tmp = t;
                if (t->left == NULL) {
                    t = t->right;
                } else {
                    if (t->right == NULL) {
                        t = t->left;
                    }
                    free(tmp);
                }
            }
        }
    }
    return t;
}

void freeTree(Tree *root) {
    if (root == NULL) {
        return;
    }
    freeTree(root->left);
    freeTree(root->right);
    free(root);
}

int numberOfSons(Tree *root) {
    if (root == NULL) {
        return 0;
    }
    return 1 + numberOfSons(root->left) + numberOfSons(root->right);
}

int rank(Tree* root) {
    int left, right;
    if (root == NULL) {
        return -1;
    }
    left = rank(root->left);
    right = rank(root->right);
    return 1 + (left > right ? left : right);
}

Tree * find(Tree * nod, int val) {
    if (nod == NULL) {
        return NULL;
    }
    if (nod->info.num == val) {
        return nod;
    }
    if (nod->info.num > val) {
        return find(nod->left, val);
    }
    return find(nod->right, val);
}

/**
 * Bug if there is no parent.
 */
Tree * findParent(Tree * nod, int val) {
    if (nod == NULL) return NULL;
    if (nod->left != NULL) {
        if (nod->left->info.num == val) {
            return nod;
        }
    }
    if (nod->right != NULL) {
        if (nod->right->info.num == val) {
            return nod;
        }
    }
    if (nod->info.num > val)return findParent(nod->left, val);
    return findParent(nod->right, val);
}

int main() {
    Tree* t = NULL;
    int i;
    Item itm;
    int arr[] = {5, 7, 6, 9, 8, 11, 10, 12, 3, 4, 1, 2, 0};
    for (i = 0; i < length(arr); i++) {
        t = add(t, arr[i]);
    }

    printInorder(t);
    printf("\nThe minimum of the tree is : %d ", findMinimum(t)->info);
    printf("\nThe amount of leaf/nodes in the tree is : %d ", numberOfSons(t));
    itm.num = 12;
    printf("\nRemoving %d from the tree.\n", itm.num);
    t = delete(t, itm);
    printInorder(t);
    if (find(t, 50) == NULL) {
        printf("\nCannot find 50.");
    }
    printf("\nThe parent of 4 is : %d.", findParent(t, 4)->info);

    freeTree(t);
    return (EXIT_SUCCESS);
}

typedef int boolean;

typedef struct {
    int num; /*! The contents of the <tt>Item</tt>.<br>Type: <tt>int</tt>*/
} Item; /*! \typedef Item
        * \struct A struct represent one item.
        */

typedef struct node {
    Item info; /*! A struct with a one <tt>int</tt> on it. */
    struct node * left; /*! A pointer to <tt>Tree</tt> struct.<br>The <b>Left</b> one.*/
    struct node * right; /*! A pointer to <tt>Tree</tt> struct.<br>The <b>Right</b> one.*/
} Tree; /*! \typedef Tree
        * \struct A struct represent one node in binary tree.<br>
        */
