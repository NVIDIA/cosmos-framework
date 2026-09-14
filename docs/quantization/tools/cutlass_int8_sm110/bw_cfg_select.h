// Expands X(...) for exactly one entry of THOR_BW_CFG_LIST (index THOR_ONLY).
#define THOR_BW_PICK(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) THOR_BW_PICK_##I(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#if THOR_ONLY == 0
#define THOR_BW_PICK_0(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_0(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 1
#define THOR_BW_PICK_1(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_1(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 2
#define THOR_BW_PICK_2(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_2(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 3
#define THOR_BW_PICK_3(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_3(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 4
#define THOR_BW_PICK_4(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_4(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 5
#define THOR_BW_PICK_5(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_5(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 6
#define THOR_BW_PICK_6(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_6(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 7
#define THOR_BW_PICK_7(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_7(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 8
#define THOR_BW_PICK_8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_8(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 9
#define THOR_BW_PICK_9(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_9(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 10
#define THOR_BW_PICK_10(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_10(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 11
#define THOR_BW_PICK_11(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_11(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 12
#define THOR_BW_PICK_12(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_12(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 13
#define THOR_BW_PICK_13(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_13(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 14
#define THOR_BW_PICK_14(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_14(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 15
#define THOR_BW_PICK_15(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_15(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 16
#define THOR_BW_PICK_16(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_16(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 17
#define THOR_BW_PICK_17(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_17(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 18
#define THOR_BW_PICK_18(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_18(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 19
#define THOR_BW_PICK_19(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_19(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 20
#define THOR_BW_PICK_20(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_20(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 21
#define THOR_BW_PICK_21(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_21(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 22
#define THOR_BW_PICK_22(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_22(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 23
#define THOR_BW_PICK_23(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_23(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 24
#define THOR_BW_PICK_24(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_24(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 25
#define THOR_BW_PICK_25(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_25(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 26
#define THOR_BW_PICK_26(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_26(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 27
#define THOR_BW_PICK_27(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_27(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 28
#define THOR_BW_PICK_28(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_28(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 29
#define THOR_BW_PICK_29(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_29(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 30
#define THOR_BW_PICK_30(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_30(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
#if THOR_ONLY == 31
#define THOR_BW_PICK_31(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D) X(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#else
#define THOR_BW_PICK_31(I, TM, TN, TK, CM, CN, S, E, GM, GN, GK, D)
#endif
THOR_BW_CFG_LIST(THOR_BW_PICK)
