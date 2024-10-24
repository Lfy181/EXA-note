# EAX-note

## 类的构造函数 `TCross` 和析构函数

### 初始化参数说明

在 `TCross::TCross(int N)` 中，参数 `N` 是旅行商问题 (TSP) 中的城市数量。以下是该构造函数内各个数据结构和变量的详细说明，以及如何使用它们来初始化和分配内存的：

### 主要参数

- **`N`**：这是传入的参数，表示旅行商问题中的城市数量。构造函数内的许多数据结构都是根据 `N` 来分配大小的。

### 成员变量和数据结构

1. **`fMaxNumOfABcycle`**：
   - **类型**：整数
   - **用途**：表示最大 AB 周期数量，设置为 2000。AB 周期是两条父代路径中的交替边形成的路径。2000 是一个通常足够处理大规模问题的默认值。

2. **`fN`**：
   - **类型**：整数
   - **用途**：表示城市的数量，它被赋值为 `N`。几乎所有数组的大小都依赖于这个值。

3. **`near_data`**：
   - 类型：二维数组 `[fN][5]`
   - **用途**：存储每个城市的邻接城市信息。每个城市可以连接到多个邻居（基于两条父代路径），因此需要一个 5 列的数组来记录不同的连接。

4. **`fABcycle`**：
   - **类型**：二维数组 `[fMaxNumOfABcycle][2 * fN + 4]`
   - **用途**：用于存储 AB 周期。AB 周期由父代路径的交替边组成，每个周期可能包含多达 2 倍的城市数，因此它是一个 2 * `fN` 长的数组，再加上额外的 4 列用于存储周期的其他信息（如周期的长度和起点/终点）。

5. **`koritsu`**：
   - **类型**：数组 `[fN]`
   - **用途**：用于在交叉过程中标记某些孤立的城市，这些城市暂时没有连到其他城市。

6. **`bunki`**：
   - **类型**：数组 `[fN]`
   - **用途**：用于存储与其他城市有分离连接的城市，这些城市在交叉过程中可能会被进一步处理。

7. **`fRoute`**：
   - **类型**：数组 `[2 * fN + 1]`
   - **用途**：表示当前路径的顺序（或路线）。在路径生成和修改过程中，该数组记录遍历的城市。

8. **`fPermu`**：
   - **类型**：数组 `[fMaxNumOfABcycle]`
   - **用途**：用于随机排列 AB 周期编号，帮助在交叉过程中随机选择周期。

9. **`fC`**：
   - **类型**：数组 `[2 * fN + 4]`
   - **用途**：一个临时数组，用于存储在计算 AB 周期和路径操作过程中生成的周期和边。

10. **`fOrder`** 和 **`fInv`**：
    - **类型**：数组 `[fN]`
    - **用途**：`fOrder` 记录城市在路径中的顺序，`fInv` 存储路径的逆序排列，用于快速查找和加速操作。

11. **`fSegment`**：
    - **类型**：二维数组 `[fN][2]`
    - **用途**：记录每个路径段的起点和终点，帮助在交叉过程中管理不同的路径段。

12. **`fGainAB`**：
    - **类型**：数组 `[fN]`
    - **用途**：用于记录每个 AB 周期的增益。增益是指通过交叉操作后改进的路径长度。

### 其他性能优化相关的结构

1. **`fModiEdge`** 和 **`fBestModiEdge`**：
   - **类型**：二维数组 `[fN][4]`
   - **用途**：用于存储需要修改的边。交叉操作过程中，路径中的边可能会被替换或调整，`fModiEdge` 记录了这些边的变化。

2. **`fAppliedCycle`** 和 **`fBestAppliedCycle`**：
   - **类型**：数组 `[fN]`
   - **用途**：用于存储在交叉过程中应用的 AB 周期编号。

### Block2 部分相关数据结构

1. **`fInEffectNode`**：
   - **类型**：二维数组 `[fN][2]` 
   - **用途**：标记哪些节点在交叉过程中是有效的节点，这些节点会受到影响。

2. **`fWeight_RR`** 和 **`fWeight_SR`**：
   - **类型**：二维数组 `[fMaxNumOfABcycle][fMaxNumOfABcycle]` 和数组 `[fMaxNumOfABcycle]`
   - **用途**：这些权重数组用于计算和评估不同的 AB 周期组合，以便找到最优的周期组合方案。

### 析构函数中释放的内存

- 构造函数中为许多二维数组和一维数组分配了内存，这些内存必须在对象销毁时释放，避免内存泄漏。析构函数 `~TCross()` 负责释放所有动态分配的内存。

### 总结

`TCross::TCross(int N)` 的核心作用是根据给定的城市数量 `N`，初始化所有与 EAX 算法相关的数据结构，并为后续的交叉操作（包括 AB 周期的生成和优化路径的寻找）做好准备。每个数据结构都围绕如何高效执行 TSP 问题的交叉过程设计，为算法的速度和内存效率提供支持。

## 核心函数 `SetParents` 和 `DoIt`

### `SetParents` 函数

`SetParents` 函数负责将两个父代（`tPa1` 和 `tPa2`）传递给交叉操作的核心函数。在交叉操作过程中，它会比较这两个父代的路径，并生成需要的 AB 周期。

### 具体作用如下：

1. **初始化交叉操作所需的结构**
   - 该函数首先调用 `SetABcycle` 函数，该函数根据父代 `tPa1` 和 `tPa2` 的路径，生成 AB 周期。AB 周期是在两条父代路径的边之间交替形成的周期，它是交叉操作中的关键部分。
   ```cpp
   this->SetABcycle( tPa1, tPa2, flagC, numOfKids );
   ```

2. **计算父代之间的路径差异**
   - 通过计算父代之间的路径差异（即父代路径中的不同边），评估交叉操作的效果。
   ```cpp
   for( int i = 0; i < fN; ++i )
   {
       // ...检查当前城市和父代之间的路径差异
       if( tPa2.fLink[ curr ][ 0 ] != next && tPa2.fLink[ curr ][ 1 ] != next )  
         ++fDis_AB; 
   }
   ```

3. **处理不同的交叉模式**
   - 通过传入的 `flagC` 参数，`SetParents` 可以根据不同的模式处理交叉操作。例如，如果 `flagC[1] == 2`，则启用 **Block2 模式**，此时会调用 `SetWeight` 函数来为不同路径段设置权重。
   ```cpp
   if( flagC[ 1 ] == 2 ){  
       fTmax = 10;
       fMaxStag = 20;
       this->SetWeight( tPa1, tPa2 );  // 为交叉操作的周期设置权重
   }
   ```

4. **路径和逆序路径的存储**
   - `fOrder` 和 `fInv` 这两个数组被设置为保存路径的顺序和逆序。交叉操作通常需要对路径进行排序或对比，因此 `fOrder` 和 `fInv` 用于帮助加速路径查找。
   ```cpp
   fOrder[ i ] = curr;
   fInv[ curr ] = i;
   ```

### 总结

`SetParents` 函数的作用是准备遗传算法的交叉操作，具体通过以下几个步骤：

1. **生成 AB 周期**：基于两个父代路径，生成用于交叉操作的 AB 周期。
2. 计算路径差异：评估父代之间的路径差异（即不同边），为生成新的解做准备。
3. **处理交叉模式**：根据不同的标志位，启用不同的交叉模式，如 Block2 模式。
4. **存储路径顺序和逆序**：通过 `fOrder` 和 `fInv` 数组，保存路径的顺序和逆序，以加速后续操作。

这些步骤为 EAX 算法中的交叉操作提供了基础，帮助生成更优的子代解。

### `DoIt`函数

`DoIt` 是 `TCross` 类中的一个核心方法，它实现了交叉操作并生成新解（子代）的过程。该函数负责在遗传算法的框架下进行路径交换和优化，通过对多个周期（AB 周期）进行组合，生成多个可能的解，并最终选择最优解。具体作用如下：

### 1. **初始化变量和设置交叉类型**
   - `DoIt` 函数首先会初始化一些用于记录解的信息，如 `Num`、`gain` 和 `pointMax` 等，准备开始生成新的子代解。
   - 通过传入的 `flagC` 参数，确定交叉类型和评价方法（例如贪心法、距离保持或熵保持等）。不同的交叉类型将影响后续如何选择 AB 周期进行组合。

   ```cpp
   fEvalType = flagC[ 0 ];   // 确定评价方式 (如贪心法、距离保持等)
   fEsetType = flagC[ 1 ];   // 确定交叉类型 (如单一 AB 周期还是 Block2)
   ```

### 2. **选择并组合 AB 周期**
   - 根据前面生成的 AB 周期，`DoIt` 函数会通过 `fPermu` 进行随机排列，或根据权重和长度选择合适的 AB 周期（通过 `SetABcycle` 函数生成）。
   - `fABcycleInEset` 用于存储当前选择的 AB 周期，准备对这些周期进行交叉操作。

   ```cpp
   fNumOfABcycleInEset = 0;
   if( fEsetType == 1 ){ // 单一 AB 周期
       jnum = fPermu[ j ];
       fABcycleInEset[ fNumOfABcycleInEset++ ] = jnum; 
   } else if( fEsetType == 2 ){ // Block2 模式
       // 选择多个 AB 周期并组合
   }
   ```

### 3. **执行交叉操作**
   - `DoIt` 通过交叉不同的 AB 周期生成新的解。每次交叉会修改当前解的路径，并计算路径增益（`gain`）以及所造成的损失（`DLoss`）。
   - `ChangeSol` 函数用于在交叉操作中修改路径，交换边来生成新的路径。

   ```cpp
   this->ChangeSol( tKid, jnum, flagP );
   gain += fGainAB[ jnum ];
   ```

### 4. **评估交叉操作的效果**
   - 交叉操作完成后，计算生成的解的性能（通过 `gain` 和 `DLoss`），并与之前的解进行比较。如果新解更优（`pointMax < point`），则记录当前最优解。

   ```cpp
   point = (double)gain / DLoss;
   if( pointMax < point ){
       pointMax = point;
       BestGain = gain;
       fFlagImp = 1;
   }
   ```

### 5. **应用最佳解**
   - 如果在当前交叉过程中找到了更优解，则调用 `GoToBest` 函数，将最优解应用到子代 `tKid` 上。
   - 另外，还会更新边的频率（`fEdgeFreq`），记录在当前生成解中哪些边被使用。

   ```cpp
   if( fFlagImp == 1 ){
       this->GoToBest( tKid );
       this->IncrementEdgeFreq( fEdgeFreq );
   }
   ```

### 6. **恢复到父代**
   - 如果当前生成的解不是最优解，则调用 `BackToPa1` 恢复到父代的路径，从而确保不保留劣解，准备下一次交叉。

   ```cpp
   this->BackToPa1( tKid );
   ```

### 总结
`DoIt` 的主要作用是在多个 AB 周期中进行交叉操作，生成新的解（子代），并通过计算路径增益和损失来选择最优解。该函数实现了遗传算法中的交叉过程，是 EAX 算法中的关键步骤，能够通过不同的评价方法（如贪心、距离或熵）优化生成解，最终提升算法解决旅行商问题（TSP）的能力。

## SetABcycle函数

`SetABcycle` 函数的主要作用是根据两个父代个体（`tPa1` 和 `tPa2`）的路径生成 **AB周期**，这是 EAX（Edge Assembly Crossover）算法中的一个核心概念。AB周期是在两个父代路径之间通过交替连接的边形成的闭合路径，它们是用于执行交叉操作生成子代的关键。

### 具体作用如下：

### 1. **初始化交叉操作所需的结构**
   `SetABcycle` 会为每一个城市生成其邻接表，表示该城市在两个父代中的连接关系。每个城市的邻接信息存储在 `near_data` 数组中。

   ```cpp
   for( int j = 0; j < fN ; ++j )
   {
     near_data[j][1]=tPa1.fLink[j][0];
     near_data[j][3]=tPa1.fLink[j][1];
     near_data[j][2]=tPa2.fLink[j][0];
     near_data[j][4]=tPa2.fLink[j][1];
   }
   ```

   - **`near_data`** 保存了每个城市在父代路径中的邻接城市。这些数据用于生成 AB 周期。

### 2. **识别并处理孤立城市**
   `SetABcycle` 通过 `koritsu` 数组来识别那些在交叉过程中暂时孤立的城市（即暂时没有连接到其他城市的城市），并将它们标记为“孤立城市”。

   ```cpp
   for(int j = 0; j < fN; ++j ) 
   {
     check_koritsu[j]=-1;
     kori_inv[koritsu[j]]=j;
   }
   ```

   - **`koritsu`** 记录孤立城市，**`kori_inv`** 用于加速对这些孤立城市的查找。

### 3. **生成 AB 周期**
   核心工作是生成 **AB 周期**，通过交替使用两个父代中的边，将城市按照一定顺序连接起来，形成一个闭合路径。这是通过一个循环完成的，函数会从一个起始城市 `st` 开始，沿着父代的边依次遍历，直到形成一个周期。

   ```cpp
   while(koritsu_many != 0)
   {
     fPosiCurr = 0;
     st = koritsu[r];
     fRoute[fPosiCurr] = st;
     ci = st;
     pr_type = 2;
     // 开始生成周期的循环...
   }
   ```

   - **`fRoute`** 记录生成的路径，通过多次迭代，不断添加路径中的城市和边。
   - **`fABcycle`** 存储生成的 AB 周期。每个 AB 周期是两个父代路径之间交替边的集合。

### 4. **处理路径的交替边**
   在生成 AB 周期时，路径中的每个节点都会根据它连接到的父代边依次选择连接到下一个节点。在形成周期的过程中，程序会不断交替使用两个父代的边，并且在某些情况下交换节点之间的连接，确保路径可以闭合。

   ```cpp
   if(near_data[ci][0] == 2)
   {
     // 检查当前城市的连接方式并交换边
     this->Swap(near_data[ci][fPosiCurr%2+1], near_data[ci][fPosiCurr%2+3]);
   }
   ```

   - **`Swap`** 函数用于在交替边时调整节点的连接，确保路径能够形成闭合周期。

### 5. **终止条件**
   一旦形成了足够的 AB 周期或者达到了设定的最大周期数（`fMaxNumOfABcycle`），函数会终止。

   ```cpp
   if( fNumOfABcycle == fMaxNumOfABcycle )
   {
     printf( "fMaxNumOfABcycle(%d) must be increased\n", fMaxNumOfABcycle );
     exit( 1 );
   }
   ```

   - **`fNumOfABcycle`** 表示已经生成的 AB 周期的数量。如果数量达到 `fMaxNumOfABcycle`，则函数终止。

### 总结
`SetABcycle` 的作用是根据父代路径生成多个 AB 周期，并存储在 `fABcycle` 中。这些周期是在两个父代路径之间交替边生成的，之后会用于交叉操作，生成新的子代解。该函数通过处理孤立城市、选择合适的边、交替连接节点的方式，逐步构建新的 AB 周期，是 EAX 算法生成解的基础。

## `FormABcycle` 函数

`FormABcycle` 函数的主要作用是在旅行商问题（TSP）的 EAX（Edge Assembly Crossover）算法中，生成一个具体的 **AB 周期**（AB cycle）。这个周期是在父代的路径之间通过交替边生成的闭合路径。AB 周期是 EAX 算法中的一个核心概念，它用于交叉操作，从而生成新的子代解。

### 具体的执行步骤和作用如下：

### 1. **初始化变量**
   - `edge_type`：用于标记当前的边类型，分为 A 边和 B 边（分别对应两个父代路径）。
   - `fC`：用于存储 AB 周期中的节点序列，作为临时数组保存遍历的城市。
   - `cem`：记录路径中城市的数量，指示周期的长度。
   - `fABcycle[fNumOfABcycle]`：最终用于存储生成的 AB 周期。

   ```cpp
   int edge_type;
   int st=fRoute[fPosiCurr];
   cem=0;
   fC[cem]=st;
   ```

### 2. **生成 AB 周期**
   - 该函数从一个起点 `st` 开始，通过交替使用 A 边和 B 边，在父代路径之间切换，直到形成一个闭合的 AB 周期。
   - **交替使用父代的边**：通过 `fPosiCurr` 变量，不断切换当前的边类型，在父代 1 和父代 2 之间交替选边。

   ```cpp
   while(1)
   {
     cem++;
     fPosiCurr--;
     ci=fRoute[fPosiCurr];
     near_data[ci][0]--;
     if(ci == st) st_count++;
     if(st_count == st_appear) break;
     fC[cem] = ci;
   }
   ```

   - 每当找到一个与起始点相同的节点，并且形成一个闭合路径时，当前的 AB 周期生成过程结束。

### 3. **存储 AB 周期**
   - AB 周期的节点存储在 `fC` 中，并在周期形成后，转移到 `fABcycle` 中，用于后续的交叉操作。每个 AB 周期还会记录头尾节点和路径长度。

   ```cpp
   for( int j=0; j<cem; j++ ) 
     fABcycle[fNumOfABcycle][j+2] = fC[j];
   ```

   - **`fABcycle[fNumOfABcycle]`** 用于存储生成的 AB 周期，其中包括路径的长度和头尾节点。路径长度存储在 `fABcycle[fNumOfABcycle][0]`，城市序列存储在 `[2]` 到 `[cem+1]`。

### 4. **计算增益**
   - 在生成 AB 周期后，函数还会计算该周期的增益，即通过该周期进行路径交换可以带来的改进。这是通过计算新路径中的距离差来实现的。

   ```cpp
   diff = 0;
   for( j = 0; j < cem/2; ++j ) 
   {
     diff = diff + eval->fEdgeDis[fC[2*j]][fC[1+2*j]] - eval->fEdgeDis[fC[1+2*j]][fC[2+2*j]];
   }
   fGainAB[fNumOfABcycle] = diff;
   ```

   - **`fGainAB[fNumOfABcycle]`** 记录每个 AB 周期的增益，用于后续的解评估和选择最优解。

### 5. **更新和检查**
   - 更新当前 AB 周期的计数，并确保生成的 AB 周期不会超过设定的最大周期数。如果达到最大值，会打印提示并退出程序。

   ```cpp
   fNumOfABcycle++;
   if(fNumOfABcycle == fMaxNumOfABcycle) {
     printf("fMaxNumOfABcycle(%d) must be increased\n", fMaxNumOfABcycle);
     exit(1);
   }
   ```

### 总结

`FormABcycle` 的作用是根据父代路径生成一个完整的 AB 周期，并将其存储在 `fABcycle` 中。通过交替使用父代边，形成闭合路径，并计算该路径的增益，为后续的交叉操作和优化提供基础。这个函数在 EAX 算法中非常重要，它生成的 AB 周期是进行路径交换和子代生成的关键。

## `ChangeSol` 函数

`ChangeSol` 函数的主要作用是根据给定的 AB 周期，修改子代个体 `tKid` 的路径，调整其中的边，生成新的路径。这个函数是 EAX 算法中的核心部分，它通过交换边来修改旅行商问题 (TSP) 的路径，从而创建新的解。

### 1. **函数输入参数**
- **`TIndi& tKid`**：表示当前的子代个体，它的路径将被修改。
- **`int ABnum`**：这是一个标识符，用于指示要应用的 AB 周期的编号。AB 周期包含交替的路径段，它们定义了哪些边将被交换。
- **`int type`**：交叉类型的标志。`type` 的值决定了如何应用 AB 周期，是直接应用还是反转后再应用。

### 2. **提取和反转 AB 周期**
   - 函数首先提取了指定的 AB 周期编号 `ABnum`，并将这个周期的数据加载到临时变量 `fC` 中。`fC` 用于存储当前周期的路径和节点。
   
   ```cpp
   cem = fABcycle[ABnum][0];
   fC[0] = fABcycle[ABnum][0];

   if (type == 2) {
       for (j = 0; j < cem + 3; j++)
           fC[cem + 3 - j] = fABcycle[ABnum][j + 1];
   } else {
       for (j = 1; j <= cem + 3; j++)
           fC[j] = fABcycle[ABnum][j];
   }
   ```

   - **`type` 参数的影响**：
     - 当 `type == 1` 时，函数直接按 AB 周期顺序修改路径。
     - 当 `type == 2` 时，周期的顺序会被反转。这意味着边的方向也会相应改变。

### 3. **交换路径中的边**
   - 核心的路径修改部分是通过交换边来完成的。对于 AB 周期中的每一对节点，函数将父代中的边替换为新的边。
   
   ```cpp
   for (j = 0; j < cem / 2; j++) {
       r1 = fC[2 + 2 * j];
       r2 = fC[3 + 2 * j];
       b1 = fC[1 + 2 * j];
       b2 = fC[4 + 2 * j];

       if (tKid.fLink[r1][0] == r2)
           tKid.fLink[r1][0] = b1;
       else
           tKid.fLink[r1][1] = b1;

       if (tKid.fLink[r2][0] == r1)
           tKid.fLink[r2][0] = b2;
       else
           tKid.fLink[r2][1] = b2;
   }
   ```

   - **`r1`, `r2`**：AB 周期中的一对节点，这对节点的边会被替换为 `b1`, `b2`。也就是说，原本连接 `r1` 和 `r2` 的边被新的边 `r1 - b1` 和 `r2 - b2` 所替代。
   - 这个过程会沿着周期进行，并在周期中每对节点之间进行边交换，从而修改整个路径。

### 4. **更新路径位置信息**
   - 函数还会更新 `fInv`（逆序数组）和 `LinkBPosi`（存储节点连接位置）等结构，用于快速查找节点在路径中的位置。这些辅助数组有助于在交叉操作后加速其他相关操作。

   ```cpp
   po_r1 = fInv[r1];
   po_r2 = fInv[r2];
   po_b1 = fInv[b1];
   po_b2 = fInv[b2];

   if (po_r1 == 0 && po_r2 == fN - 1)
       fSegPosiList[fNumOfSPL++] = po_r1;
   else if (po_r1 == fN - 1 && po_r2 == 0)
       fSegPosiList[fNumOfSPL++] = po_r2;
   ```

   - **`fInv`**：存储节点在路径中的位置，方便后续操作查找节点的具体位置。
   - **`fSegPosiList`**：用于记录段的位置，有助于在生成新的路径段时快速找到需要调整的部分。

### 5. **更新辅助结构**
   - 修改完成后，函数更新一些辅助结构，如 `LinkBPosi`，它保存了新边在路径中的位置。这些结构在后续的解评估和进一步操作中非常有用。

   ```cpp
   LinkBPosi[po_r1][1] = LinkBPosi[po_r1][0];
   LinkBPosi[po_r2][1] = LinkBPosi[po_r2][0];
   LinkBPosi[po_r1][0] = po_b1;
   LinkBPosi[po_r2][0] = po_b2;
   ```

### 总结
`ChangeSol` 的作用是根据指定的 AB 周期，对子代个体 `tKid` 的路径进行修改。它通过交换周期中的边，生成新的路径解。该函数是 EAX 算法中的核心部分，通过改变父代解的部分路径，从而生成新的、可能更优的子代解。

## `MakeCompleteSol` 函数

`MakeCompleteSol` 函数的主要作用是在交叉操作后，将未完成的部分路径补全，生成一个完整的旅行商问题（TSP）解。它在 `ChangeSol` 函数之后运行，用于修复路径中的不完整段，确保每个城市都被访问且形成一个闭合的路径。这是 EAX 算法中的重要步骤，用于保证生成的子代是一个有效的 TSP 解。

### 1. **初始化**
函数首先初始化一些变量，包括路径段的计数、当前路径的起点、终点等。

```cpp
fGainModi = 0;
min_unit_city = fN + 12345;
```

- **`fGainModi`**：初始化为 0，用于存储通过修复路径而获得的增益。
- **`min_unit_city`**：记录路径中最小的单位城市数。用于找到需要处理的路径段。

### 2. **查找中心路径段**
函数通过遍历所有路径段，查找中心路径段（即节点数最少的路径段）。中心路径段将在后续修复过程中作为处理的起点。

```cpp
for (int u = 0; u < fNumOfUnit; ++u) {
    if (fNumOfElementInUnit[u] < min_unit_city) {
        center_un = u;
        min_unit_city = fNumOfElementInUnit[u];
    }
}
```

- **`center_un`**：记录中心路径段的索引。后续操作会围绕该路径段展开。

### 3. **遍历并构建完整路径**
接下来，函数通过 `curr` 和 `next` 等变量遍历当前路径段中的所有城市，构建路径的完整序列。

```cpp
curr = -1;
next = st;
while (1) {
    pre = curr;
    curr = next;
    fCenterUnit[curr] = 1;
    fListOfCenterUnit[fNumOfElementInCU++] = curr;

    if (tKid.fLink[curr][0] != pre)
        next = tKid.fLink[curr][0];
    else
        next = tKid.fLink[curr][1];

    if (next == st) break;
}
```

- **`fCenterUnit`**：用于标记路径段中哪些城市属于中心段。
- **`fListOfCenterUnit`**：存储该路径段中的所有城市，以便后续操作。

### 4. **计算增益**
在找到完整路径段后，函数通过计算增益（`gain`）来优化路径。通过交换边和调整节点位置，函数能够找到增益最大的调整方案，从而生成最优的路径。

```cpp
for (int s = 1; s <= fNumOfElementInCU; ++s) {
    // 尝试找到增益最大的路径调整
    if (fCenterUnit[c] == 0) {
        diff = eval->fEdgeDis[a][b] + eval->fEdgeDis[c][d] - eval->fEdgeDis[a][c] - eval->fEdgeDis[b][d];
        if (diff > max_diff) {
            // 更新最优调整方案
        }
    }
}
```

- **`eval->fEdgeDis`**：用于计算城市之间的距离差异，从而评估路径调整的增益。

### 5. **修复路径并更新边**
根据计算得到的最优增益方案，函数通过交换边修复路径，生成完整的解。

```cpp
if (tKid.fLink[aa][0] == bb)
    tKid.fLink[aa][0] = a1;
else
    tKid.fLink[aa][1] = a1;

// 更新其他边
```

- **`tKid.fLink`**：存储路径中的边，修复路径时通过调整这些边生成新的路径。

### 6. **更新段信息**
修复完成后，函数会更新段信息，确保路径段的完整性。

```cpp
for (int s = 0; s < fNumOfSeg; ++s) {
    if (fSegUnit[s] == select_un)
        fSegUnit[s] = center_un;
}
```

- **`fSegUnit`**：用于记录每个路径段所属的单位。

### 总结
`MakeCompleteSol` 的作用是在交叉操作后，修复路径中的不完整段，生成一个有效的旅行商问题解。它通过计算路径调整的增益，选择最优的边交换方式，确保生成的子代解是一个完整、有效的 TSP 解。

## `MakeUnit` 函数

`MakeUnit` 函数的主要作用是根据现有的路径段信息，将多个路径段组合成单元（unit），并将每个单元内的城市合理分配到相应的路径段中，从而为进一步的路径优化和调整做好准备。这个步骤是为了确保路径段的完整性，使路径中的城市和边能够被正确地组织和处理。

### 1. **生成初始单元列表**
   首先，`MakeUnit` 函数会扫描路径段，并将路径段初始化为单独的单元。如果路径段中没有已经处理过的城市，它会将该段标记为一个新的单元。

   ```cpp
   int flag = 1;
   for (int s = 0; s < fNumOfSPL; ++s) {
       if (fSegPosiList[s] == 0) {
           flag = 0;
           break;
       }
   }
   ```

   - **`fSegPosiList`**：存储路径段的位置，它包含了每个路径段的起始和终止位置。
   - **`fNumOfSPL`**：表示当前路径段的数量。

### 2. **分割路径并分配单元**
   接下来，函数会对路径段进行分割，将相邻的城市和边组合成单元。每个路径段的起点和终点会被标记为同一个单元。函数遍历所有路径段，将未被处理的段分配到新的单元中。

   ```cpp
   for (int s = 0; s < fNumOfSeg - 1; ++s) {
       fSegment[s][0] = fSegPosiList[s];
       fSegment[s][1] = fSegPosiList[s + 1] - 1;
   }
   ```

   - **`fSegment`**：存储每个路径段的起点和终点，用于定义路径段的范围。
   - **`fNumOfSeg`**：路径段的数量。

### 3. **建立城市到单元的映射**
   每个单元中的城市和路径段被映射到相应的单元号，确保路径段中的城市被正确分配。通过 `LinkAPosi` 和 `LinkBPosi` 数组，函数建立了每个城市到路径段的映射。

   ```cpp
   for (int s = 0; s < fNumOfSeg; ++s) {
       LinkAPosi[fSegment[s][0]] = fSegment[s][1];
       LinkAPosi[fSegment[s][1]] = fSegment[s][0];
       fPosiSeg[fSegment[s][0]] = s;
       fPosiSeg[fSegment[s][1]] = s;
   }
   ```

   - **`LinkAPosi` 和 `LinkBPosi`**：用于存储每个城市的前后连接节点，用于路径段之间的连接操作。
   - **`fPosiSeg`**：用于记录每个城市属于哪个路径段。

### 4. **分配单元编号**
   `MakeUnit` 会将每个路径段分配到单元中。每个单元的编号通过 `fSegUnit` 记录，函数还会计算每个单元中包含的城市数量。

   ```cpp
   for (int s = 0; s < fNumOfSeg; ++s) {
       fSegUnit[s] = -1;
   }
   ```

   - **`fSegUnit`**：用于记录每个路径段所属的单元编号。
   - **`fNumOfUnit`**：记录生成的单元数量。

### 5. **优化单元结构**
   生成初始单元后，函数会合并相邻的单元，并确保每个单元的城市数量和边结构保持合理。单元的数量会在此过程中减少，路径结构变得更加紧凑。

   ```cpp
   for (int s = 0; s < fNumOfSeg; ++s) {
       fNumOfElementInUnit[unitNum] += fSegment[s][1] - fSegment[s][0] + 1;
   }
   ```

   - **`fNumOfElementInUnit`**：存储每个单元中的城市数量。

### 总结
`MakeUnit` 函数的作用是将路径分割为多个单元，并根据城市和路径段的连接关系进行合理分配。通过将相邻路径段组合成单元，函数为后续的路径优化和调整操作做好了准备。这一过程确保了路径结构的完整性，并为进一步的边交换和优化操作提供了坚实基础。


## `BackToPa1`函数

`BackToPa1` 函数的主要作用是将当前解 `tKid` 的路径恢复到父代 `Pa1` 的原始状态。该函数在生成新的路径解时会尝试多个 AB 周期进行修改，如果发现新的路径不优或需要重置，`BackToPa1` 函数会将路径恢复到原来的父代解。这一操作在交叉操作中至关重要，因为它确保了在不保留劣质解的情况下，能够多次尝试不同的路径修改。

### 具体作用如下：

### 1. **恢复路径中的边**
   函数通过逆转之前在路径修改过程中应用的边，逐步将路径恢复到父代 `Pa1` 的状态。它会逐一遍历之前修改的边，并通过存储的 `fModiEdge` 结构将它们恢复为原始的父代边。

   ```cpp
   for (int s = fNumOfModiEdge -1; s >= 0; --s ) {
       aa = fModiEdge[s][0];
       a1 = fModiEdge[s][1];
       bb = fModiEdge[s][2];
       b1 = fModiEdge[s][3];

       // 恢复边
       if (tKid.fLink[aa][0] == bb) tKid.fLink[aa][0] = a1;
       else tKid.fLink[aa][1] = a1;
       if (tKid.fLink[b1][0] == a1) tKid.fLink[b1][0] = bb;
       else tKid.fLink[b1][1] = bb;
   }
   ```

   - **`fModiEdge`**：存储了之前在路径修改过程中改变的边。每个 `fModiEdge` 包含 4 个元素（`aa`, `a1`, `bb`, `b1`），表示在交叉过程中被替换的边。
   - **`tKid.fLink`**：存储了当前子代 `tKid` 的路径，其中的边连接信息会通过这个结构恢复。

### 2. **还原应用的 AB 周期**
   在恢复边之后，函数还会依次逆转之前应用的 AB 周期，确保路径完全回到父代的状态。每次恢复周期时，调用 `ChangeSol` 函数将路径改回应用之前的状态。

   ```cpp
   for (int s = 0; s < fNumOfAppliedCycle; ++s) {
       jnum = fAppliedCylce[s];
       this->ChangeSol(tKid, jnum, 2);
   }
   ```

   - **`fAppliedCycle`**：存储了之前应用的 AB 周期的编号。通过遍历这些编号，`BackToPa1` 函数可以依次将所有 AB 周期的修改撤销。
   - **`ChangeSol`**：这是一个辅助函数，用于根据 AB 周期修改路径。在这里，`ChangeSol` 被用于恢复原始状态，将之前应用的 AB 周期逆转。

### 3. **恢复父代的完整路径**
   在恢复了所有的修改之后，`BackToPa1` 函数最终会让 `tKid` 的路径与 `Pa1` 完全一致。通过撤销所有的边修改和 AB 周期应用，函数确保交叉操作不会保留不优解。

### 4. **总结**
`BackToPa1` 函数的主要功能是撤销之前对路径的所有修改，将路径恢复到父代的原始状态。这一功能在遗传算法中的交叉操作过程中尤为重要，因为它允许算法在多个不同的 AB 周期之间进行尝试，确保只有最优的解才会被保留，而其他劣解能够被及时撤销。

## `IncrementEdgeFreq` 函数

`IncrementEdgeFreq` 函数的主要作用是更新边频率矩阵 `fEdgeFreq`，该矩阵用于记录在交叉操作过程中，每条边被使用的次数。在遗传算法的交叉操作过程中，边的频率信息非常重要，它可以用于评价和选择路径，从而保持种群多样性和路径结构的优化。

### 具体作用如下：

### 1. **遍历所有应用的 AB 周期**
   函数首先会遍历所有已经应用的 AB 周期，并从每个周期中提取城市对和对应的边。每个 AB 周期表示一组交替的边，通过提取这些边，函数可以对当前解中的边进行频率统计。

   ```cpp
   for (int s = 0; s < fNumOfBestAppliedCycle; ++s) {
       jnum = fBestAppliedCylce[s];
       cem = fABcycle[jnum][0];
       fC[0] = fABcycle[jnum][0];
   ```

   - **`fBestAppliedCycle`**：存储当前子代解中应用的最佳 AB 周期的编号。
   - **`fABcycle`**：存储 AB 周期的具体信息，包括路径长度、边和节点。

### 2. **更新边的频率**
   对于每一个 AB 周期，函数遍历其中的每一对节点（即每一条边），并根据它们的连接关系更新频率矩阵 `fEdgeFreq`。如果一条边出现在当前解中，它的频率会增加；如果一条边被删除，频率则会减少。

   ```cpp
   for (j = 0; j < cem / 2; ++j) {
       r1 = fC[2 + 2 * j]; r2 = fC[3 + 2 * j];
       b1 = fC[1 + 2 * j]; b2 = fC[4 + 2 * j];

       ++fEdgeFreq[r1][b1];  // 增加边 r1-b1 的频率
       --fEdgeFreq[r1][r2];  // 减少边 r1-r2 的频率
       ++fEdgeFreq[r2][b2];  // 增加边 r2-b2 的频率
   }
   ```

   - **`fEdgeFreq`**：这是一个二维数组，存储城市之间边的使用频率。通过对这些边的频率进行更新，算法能够记录哪些边在解的生成过程中被多次使用，从而帮助优化后续的交叉操作。

### 3. **更新修改的边**
   除了应用的 AB 周期，函数还会处理那些由于修改而添加或删除的边。它遍历存储的修改边列表 `fBestModiEdge`，根据其包含的节点对来更新频率。

   ```cpp
   for (int s = 0; s < fNumOfBestModiEdge; ++s) {
       aa = fBestModiEdge[s][0];
       bb = fBestModiEdge[s][1];
       a1 = fBestModiEdge[s][2];
       b1 = fBestModiEdge[s][3];

       ++fEdgeFreq[aa][a1];
       ++fEdgeFreq[bb][b1];
       --fEdgeFreq[aa][bb];
       --fEdgeFreq[a1][b1];
   }
   ```

   - **`fBestModiEdge`**：这是在路径修改过程中被应用的边的列表，每个条目代表一组已经修改的边，通过这些边的信息来更新 `fEdgeFreq`。

### 4. **总结**
`IncrementEdgeFreq` 函数的作用是在交叉操作和路径修改后，更新边的使用频率。它通过统计 AB 周期中的边和修改的边来调整边的频率矩阵，这对于优化交叉操作、保持种群多样性以及选择最优解至关重要。通过频率统计，算法可以更好地控制解的多样性，避免过度使用某些特定边。


## `InitURandom()` 函数

在你提供的代码中，`rand.cpp` 实现了一个随机数生成器类 `TRandom`，并使用了 C++ 标准库的 **Mersenne Twister**（`std::mt19937`）作为随机数生成器。以下是对代码中不同部分的详细解释：

### 1. **`InitURandom()` 函数**
   - **作用**：初始化全局随机数生成器 `rng`，并使用当前时间（`time(0)`）作为种子，这保证了每次程序运行时的随机性不同。
   - **使用场景**：在程序开始时调用该函数，确保随机数生成器已正确初始化。
   - **重载版本**：还有一个重载的 `InitURandom(int dd)` 函数，可以用自定义的整数值 `dd` 作为种子。这对调试非常有用，因为指定种子可以让随机数序列在不同运行中保持一致。

### 2. **`TRandom::Integer(int minNumber, int maxNumber)`**
   - **作用**：生成一个在 `minNumber` 和 `maxNumber` 范围内的随机整数。使用 `std::uniform_int_distribution` 实现。
   - **使用场景**：可以用于随机选择一个离散的整数值，例如在路径优化过程中随机选择城市。

### 3. **`TRandom::Double(double minNumber, double maxNumber)`**
   - **作用**：生成一个在 `minNumber` 和 `maxNumber` 之间的随机浮点数。使用 `std::uniform_real_distribution` 实现。
   - **使用场景**：需要生成连续的随机数时使用，例如概率选择。

### 4. **`TRandom::Permutation(int *array, int numOfElement, int numOfSample)`**
   - **作用**：生成一个数组的随机排列，选择指定数量的元素（`numOfSample`）。
   - **使用场景**：可以用于遗传算法中生成随机的个体或解决方案，或者在交叉和变异操作中打乱顺序。

### 5. **`TRandom::NormalDistribution(double mu, double sigma)`**
   - **作用**：生成服从正态分布的随机数，均值为 `mu`，标准差为 `sigma`。使用 `std::normal_distribution` 实现。
   - **使用场景**：在某些问题中，生成服从正态分布的随机数可以更接近实际情况，如模拟带噪声的环境或生成近似解。

### 6. **`TRandom::Shuffle(int *array, int numOfElement)`**
   - **作用**：对传入的数组进行随机打乱，使用 `std::shuffle` 和 Mersenne Twister 引擎。
   - **使用场景**：可以用于随机打乱一个路径或个体，用于遗传算法中的交叉操作。

### **总结**
- **输入**：该模块并不直接从用户处获取输入，所有输入都来自函数参数，比如要生成的随机数范围或数组。
- **输出**：生成随机数、打乱数组、生成随机排列等功能作为输出，供遗传算法或其他优化算法使用。

这个 `TRandom` 类是你程序中生成随机数和处理随机操作的核心部分，确保了算法的多样性和随机性。





## Opt算法

Opt算法：2-opt，3-opt，Or-opt，k-opt-CSDN博客
https://blog.csdn.net/sinat_41348401/article/details/126920506

这段代码实现的是一个基于2-opt算法的路径优化问题，可能是用于解决类似**旅行商问题（TSP，Travelling Salesman Problem）**的场景。算法通过不断反转路径中的部分段落，来寻找更短的路径。

让我逐行解释代码的功能和逻辑。

### 全局变量

```python
MAXCOUNT = 10
```
这个常量控制算法在无法继续优化时，最多允许的连续失败次数。如果路径经过10次尝试仍未找到更好的路径，则算法停止。

### 距离计算函数

```python
def calDist(xid, yid, durationlist):
    for dura in durationlist:
        if dura['origin'] == xid and dura['destination'] == yid:
            return dura['duration']
```

#### 功能：
`calDist`函数用于计算从一个地点 `xid` 到另一个地点 `yid` 的时间或距离。它通过在 `durationlist`（包含地点之间的时长数据）中查找匹配的原点和目的地，返回它们之间的时长。

#### 可能的问题：
如果在 `durationlist` 中找不到对应的时间，函数将返回 `None`，这可能会在后续计算中导致问题。因此可以考虑在找不到数据时返回0或其它默认值。

---

```python
def calPathDist(LocidList, durationlist):
    sum = 0
    for i in range(1, len(LocidList)):
        sum += calDist(LocidList[i], LocidList[i - 1], durationlist)
    return sum
```

#### 功能：
`calPathDist` 函数用于计算一条路径的总距离。`LocidList` 包含的是一系列地点的ID，函数通过调用 `calDist` 来获取每两个相邻地点之间的距离，并累计起来，最终返回路径的总长度。

---

### 路径比较函数

```python
def pathCompare(path1, path2, LocidList, durationlist):
    LocidList1 = []; LocidList2 = []
    for i in path1:
        LocidList1.append(LocidList[i])
    for i in path2:
        LocidList2.append(LocidList[i])
    if calPathDist(LocidList1, durationlist) <= calPathDist(LocidList2, durationlist):
        return True
    return False
```

#### 功能：
`pathCompare` 函数比较两条路径的总长度。`path1` 和 `path2` 是路径的索引，`LocidList` 是所有地点的位置列表，`durationlist` 是各地点之间的距离。函数先把路径的索引转换成地点的列表，然后分别计算两条路径的距离，如果 `path1` 的距离小于或等于 `path2`，则返回 `True`。

#### 解释：
这是2-opt算法的关键部分，用来判断路径经过优化后是否更短。

---

### 随机路径生成函数

```python
def generateRandomPath(bestPath):
    a = np.random.randint(len(bestPath))
    while True:
        b = np.random.randint(len(bestPath))
        if np.abs(a - b) > 1:
            break
    if a > b:
        return b, a, bestPath[b:a + 1]
    else:
        return a, b, bestPath[a:b + 1]
```

#### 功能：
`generateRandomPath` 函数从路径 `bestPath` 中随机选取一个子路径段，并返回该子路径段的起点和终点索引。

#### 解释：
- 首先，随机生成一个索引 `a`，表示子路径的起点。
- 然后，再随机生成一个索引 `b`，并确保 `b` 和 `a` 之间至少相差2（保证路径反转有意义），避免路径段过短。
- 最后返回 `a` 和 `b` 以及从 `bestPath[a]` 到 `bestPath[b]` 之间的子路径。

---

### 路径反转函数

```python
def reversePath(path):
    rePath = path.copy()
    rePath = reversed(rePath)
    return list(rePath)
```

#### 功能：
`reversePath` 函数用于反转给定的路径。

#### 解释：
这个函数接收一个路径列表，将其反转并返回新的反转路径。

---

### 更新最佳路径

```python
def updateBestPath(bestPath, LocidList, durationlist):
    count = 0
    while count < MAXCOUNT:
        start, end, path = generateRandomPath(bestPath)
        rePath = reversePath(path)
        if pathCompare(path, rePath, LocidList, durationlist):
            count += 1
            continue
        else:
            count = 0
            bestPath[start:end+1] = rePath
    return bestPath
```

#### 功能：
`updateBestPath` 是核心优化算法。通过反复生成随机路径段并进行反转，然后比较原始路径和反转后的路径，看看是否能得到更短的路径。如果找到了更短的路径，就更新当前的最佳路径 `bestPath`。

#### 解释：
- 如果新反转的路径比原路径更短，程序会重置 `count`，并将原路径替换为新的反转路径。
- 如果连续 `MAXCOUNT` 次未找到更好的路径，优化过程终止，返回最终的最佳路径。

---

### 输入数据生成函数

```python
def ProduceInput(routes):
    routes_pathindex = np.arange(0, len(routes))
    test = mysql()
    orderlist = test.findALLorder()
    duration = test.findALLduration()
    location = test.findAllloc()
    orderlist_route = []
    for order in orderlist:
        if str(order['OrderId']) in routes:
            orderlist_route.append(order)
    LocId = []
    for order in orderlist_route:
        LocId.append(test.findloc(order['Lat'], order['Lng']))
    return routes_pathindex, LocId, duration
```

#### 功能：
`ProduceInput` 函数从数据库中获取订单、位置信息以及两点之间的时间/距离信息。

#### 解释：
- 它首先初始化一个 `mysql()` 类，假设这个类封装了数据库操作。
- 然后从数据库中读取所有订单、时长和位置数据。
- 根据提供的 `routes` 列表筛选出与之相关的订单，并生成 `LocId` 列表，包含路径中每个点的地理位置信息。

---

### 主程序

```python
if __name__ == '__main__':
    routes = ['40001', '40003', '40002', '40004', '40005', '40006', '40007', '40008']
    (routes_pathindex, LocId, duration) = ProduceInput(routes)
    print(updateBestPath(routes_pathindex, LocId, duration))
```

#### 功能：
主程序首先定义了路径 `routes`，然后调用 `ProduceInput` 函数生成路径索引、地点位置和距离数据，最终调用 `updateBestPath` 来优化路径并输出结果。

### 总结：
- 这段代码实现了基于2-opt的路径优化算法，核心逻辑是通过不断反转随机子路径来寻找更短的路径。
- 函数设计合理，整体思路清晰，但在细节上可以优化，比如距离计算的空值处理，函数间的代码复用和简化路径操作。


## 

### 这套代码吃透。一：要知道程序的入口和出口，比如说，坐标数据输入是从哪里输入，后面EAX算法的流程： 1)生成两个初始解；2）得到AB-cycles；3）得到E-sets；4）得到中间解；5）得到最终解（下一代的解）这几步，分别是在程序的哪些部分实现的；二：要从自己的角度分析，它每一步这样做的理由，算法为什么具有好性能的理由。

### 一：程序的入口和出口


根据你提供的主程序代码，可以分析出程序的输入和输出端口如下：

### **输入端口**：

1. **命令行参数 (`argv[]`)**：
   - `argv[1]`: 这是一个整数，表示程序运行的最大试验次数 `maxNumOfTrial`。程序通过 `sscanf(argv[1], "%d", &maxNumOfTrial)` 来读取它。
   - `argv[2]`: 这是一个字符串，表示输出文件的名称 `dstFile`，用于保存程序运行结果。
   - `argv[3]`: 这是一个整数，表示种群的大小 `fNumOfPop`，通过 `sscanf(argv[3], "%d", &d)` 读取并存储到 `gEnv->fNumOfPop`。
   - `argv[4]`: 这是一个整数，表示每次迭代中生成的子代个体数 `fNumOfKids`，通过 `sscanf(argv[4], "%d", &d)` 读取并存储到 `gEnv->fNumOfKids`。
   - `argv[5]`: 这是一个字符串，表示 TSP 问题的输入文件名 `fFileNameTSP`，用于加载旅行商问题实例数据。
   - `argv[6]`（可选）：这是一个字符串，表示初始种群的输入文件名 `fFileNameInitPop`，用于加载已有的种群数据。如果没有提供此参数，则种群将会随机初始化。

### **输出端口**：

1. **`gEnv->PrintOn(n, dstFile)`**：
   - 该函数负责将当前迭代 `n` 的结果输出到指定的文件 `dstFile`。这个函数会打印一些运行时的统计信息或者解的质量。

2. **`gEnv->WriteBest(dstFile)`**：
   - 该函数会将当前找到的最优解写入输出文件 `dstFile`。程序会在每次试验后调用这个函数，将找到的最优解保存到指定的文件中。

### **总体输入输出流程**：

1. **输入**：
   - 程序从命令行参数读取试验次数、输出文件名、种群大小、子代个体数、TSP 问题文件，以及（可选的）初始种群文件。

2. **输出**：
   - 程序通过输出文件 `dstFile` 记录每次迭代的运行信息和最优解，确保结果在每次试验后被正确保存。
  
### 分析：
```bash
g++ -o jikken -O3 main.cpp env.cpp cross.cpp evaluator.cpp indi.cpp rand.cpp kopt.cpp sort.cpp -lm
```

- **`g++`**：这是 GNU 编译器（GCC）的 C++ 语言编译器，用于编译 C++ 代码。
- **`-o jikken`**：这是输出选项，`-o` 指定生成的可执行文件名为 `jikken`。编译成功后，会生成名为 `jikken` 的可执行文件。
- **`-O3`**：这是优化选项。`-O3` 表示开启最高级别的优化，编译器会尽量优化生成的代码以提高性能。虽然优化会让程序运行得更快，但编译时间也会更长。
- **`main.cpp env.cpp cross.cpp evaluator.cpp indi.cpp rand.cpp kopt.cpp sort.cpp`**：这些是需要编译的源代码文件。`g++` 会将这些 `.cpp` 文件编译，并将其链接在一起生成可执行文件。
  - `main.cpp`：通常是程序的主文件，包含 `main` 函数。
  - 其他 `.cpp` 文件：这些是项目的其他模块，它们可能包含环境设置、交叉算法、评估函数、个体表示等功能。
- **`-lm`**：链接数学库。`-lm` 是告诉编译器链接数学库 `libm`，以便使用数学函数（如 `sin`, `cos`, `sqrt` 等）。

### 总结：
这个命令将多个 `.cpp` 文件编译为一个名为 `jikken` 的可执行文件，并且应用最高级别的代码优化（`-O3`），并且链接了数学库 `libm`。



