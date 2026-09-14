// Expands X(...) for exactly one entry of THOR_CFG_LIST (index THOR_ONLY). Included by the generated cfg_*.cu files.
#define THOR_PICK(I, TM, TN, TK, CM, CN, S, E, D) THOR_PICK_##I(I, TM, TN, TK, CM, CN, S, E, D)
#if THOR_ONLY == 0
#define THOR_PICK_0(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_0(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 1
#define THOR_PICK_1(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_1(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 2
#define THOR_PICK_2(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_2(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 3
#define THOR_PICK_3(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_3(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 4
#define THOR_PICK_4(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_4(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 5
#define THOR_PICK_5(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_5(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 6
#define THOR_PICK_6(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_6(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 7
#define THOR_PICK_7(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_7(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 8
#define THOR_PICK_8(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_8(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 9
#define THOR_PICK_9(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_9(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 10
#define THOR_PICK_10(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_10(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 11
#define THOR_PICK_11(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_11(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 12
#define THOR_PICK_12(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_12(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 13
#define THOR_PICK_13(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_13(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 14
#define THOR_PICK_14(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_14(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 15
#define THOR_PICK_15(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_15(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 16
#define THOR_PICK_16(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_16(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 17
#define THOR_PICK_17(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_17(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 18
#define THOR_PICK_18(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_18(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 19
#define THOR_PICK_19(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_19(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 20
#define THOR_PICK_20(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_20(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 21
#define THOR_PICK_21(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_21(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 22
#define THOR_PICK_22(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_22(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 23
#define THOR_PICK_23(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_23(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 24
#define THOR_PICK_24(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_24(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 25
#define THOR_PICK_25(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_25(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 26
#define THOR_PICK_26(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_26(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 27
#define THOR_PICK_27(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_27(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 28
#define THOR_PICK_28(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_28(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 29
#define THOR_PICK_29(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_29(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 30
#define THOR_PICK_30(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_30(I, TM, TN, TK, CM, CN, S, E, D)
#endif
#if THOR_ONLY == 31
#define THOR_PICK_31(I, TM, TN, TK, CM, CN, S, E, D) X(I, TM, TN, TK, CM, CN, S, E, D)
#else
#define THOR_PICK_31(I, TM, TN, TK, CM, CN, S, E, D)
#endif
THOR_CFG_LIST(THOR_PICK)
