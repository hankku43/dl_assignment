<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';

// 狀態管理
const isDark = ref(true);
const articleTitle = ref('');
const prediction = ref('');
const selectedLabel = ref('');
const message = ref('');
const isLoading = ref(false);

const themes = ["Baseball", "Boy-Girl", "C_Chat", "HatePolitics", "Lifeismoney", "Military", "PC_Shopping", "Stock", "Tech_Job"];

const toggleDarkMode = () => {
  isDark.value = !isDark.value;
  localStorage.setItem('user-theme', isDark.value ? 'dark' : 'light');
};

onMounted(() => {
  const saved = localStorage.getItem('user-theme');
  isDark.value = saved ? saved === 'dark' : true;
});

const getPrediction = async () => {
  if (!articleTitle.value) return;
  isLoading.value = true;
  prediction.value = '';

  try {
    // 這裡的 URL 請根據你的 FastAPI 埠號調整
    const res = await axios.get('/api/model/predict', { params: { title: articleTitle.value } });
    prediction.value = res.data.prediction;
    selectedLabel.value = res.data.prediction;
  } catch (err) {
    alert('無法連線至後端服務，請確認 FastAPI 是否已啟動。');
  } finally {
    isLoading.value = false;
  }
};

const submitFeedback = async () => {
  try {
    await axios.post('/api/model/feedback', {
      label: selectedLabel.value,
      article_title: articleTitle.value
    });
    message.value = '✓ 感謝您的回饋，模型已記錄';
    setTimeout(() => {
      message.value = '';
      prediction.value = '';
      articleTitle.value = '';
    }, 2500);
  } catch (err) {
    alert('儲存回饋失敗');
  }
};
</script>

<template>
  <div :class="{ 'dark-theme': isDark }" class="theme-wrapper">

    <!-- 優化後的旋轉背景 -->
    <div class="blob-container">
      <!-- 亮色模式使用淺米色/淺桃色，深色模式切換為琥珀色 -->
      <div class="blob w-150 h-150 bg-orange-200 dark:bg-amber-900 top-0 left-0 animate-pulse-slow"></div>
      <div class="blob w-120 h-120 bg-amber-100 dark:bg-orange-950 bottom-10 right-10 animate-pulse-slow"
        style="animation-delay: -5s"></div>
      <div class="blob w-100 h-100 bg-stone-200 dark:bg-stone-800 top-1/2 left-1/3 animate-pulse-slow"
        style="animation-delay: -10s"></div>
      <!-- 增加一個白色點來稀釋色彩，避免髒感 -->
      <div class="blob w-180 h-180 bg-white dark:bg-transparent top-1/4 right-1/4 opacity-20"></div>
    </div>
    <div class="relative z-10 flex items-center justify-center min-h-screen p-6">

      <!-- 調整圓角與內距 -->
      <div class="max-w-xl w-full p-10 md:p-14 rounded-[3rem] glass-card">

        <div class="flex justify-between items-center mb-12">
          <div
            class="flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/5 dark:bg-emerald-500/10 border border-emerald-500/10">
            <div class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
            <span class="text-[9px] font-black tracking-widest text-emerald-600 dark:text-emerald-500 uppercase">Neural
              Engine Active</span>
          </div>
          <button @click="toggleDarkMode"
            class="px-4 py-2 rounded-full bg-stone-100 dark:bg-white/5 border border-stone-200 dark:border-white/10 hover:bg-amber-500 hover:text-white transition-all cursor-pointer">
            <span class="text-[9px] font-black tracking-widest uppercase">
              {{ isDark ? 'Dark Mode' : 'Light Mode' }}
            </span>
          </button>
        </div>

        <header class="text-center mb-12">
          <h1 class="text-4xl font-black tracking-tighter text-amber-600 dark:text-amber-500 mb-4">
            AI 主題辨識系統
          </h1>
        </header>

        <div class="space-y-6">
          <div class="relative group">
            <div
              class="absolute -top-2 left-6 px-2 text-[8px] font-black text-amber-600 bg-white dark:bg-stone-900 z-20">
              ARTICLE TITLE
            </div>
            <!-- 修改輸入框：增加陰影與更細的邊框 -->
            <input v-model="articleTitle" placeholder="貼上標題內容..."
              class="w-full px-8 py-5 rounded-3xl outline-none border border-stone-200 dark:border-stone-800 focus:border-amber-500 focus:ring-4 focus:ring-amber-500/5 transition-all text-base shadow-sm"
              style="background-color: var(--input-bg); color: var(--app-text);" />
          </div>

          <button @click="getPrediction" :disabled="!articleTitle || isLoading"
            class="w-full bg-amber-600 hover:bg-amber-700 text-white font-black py-5 rounded-3xl tracking-widest text-xs shadow-lg shadow-amber-600/20 active:scale-95 disabled:opacity-20 transition-all cursor-pointer">
            {{ isLoading ? 'ANALYZING...' : 'RUN IDENTIFICATION' }}
          </button>
        </div>

        <!-- 結果顯示區：優化層次 -->
        <transition name="pop">
          <div v-if="prediction" class="mt-12">
            <!-- 這裡使用 var(--result-bg) 讓它在深淺色自動切換背景 -->
            <div class="rounded-[2.8rem] p-10 text-center border shadow-xl backdrop-blur-md"
              :style="{ backgroundColor: 'var(--result-bg)', borderColor: 'var(--result-border)' }">
              <p class="text-[9px] font-black tracking-[0.5em] text-amber-600/50 mb-4 uppercase">Prediction</p>

              <!-- 套用 .result-text 類別 -->
              <h2 class="text-6xl font-black tracking-tighter mb-10 result-text">
                {{ prediction }}
              </h2>

              <div class="pt-8 border-t border-stone-200 dark:border-white/10">
                <p class="text-[9px] font-bold text-stone-400 dark:text-stone-500 mb-5 uppercase tracking-widest">
                  Correction Feedback</p>
                <div class="flex gap-2">
                  <!-- 下拉選單：加入 dark: 樣式確保在深色模式下也是黑色的 -->
                  <select v-model="selectedLabel" style="background-color: var(--input-bg); color: var(--app-text);"
                    class="flex-1 px-4 py-3 rounded-2xl bg-white dark:bg-stone-800 text-xs font-bold outline-none border border-stone-100 dark:border-none shadow-sm">
                    <option v-for="t in themes" :key="t" :value="t">{{ t }}</option>
                  </select>
                  <button @click="submitFeedback"
                    class="bg-amber-600 text-white px-8 py-3 rounded-2xl text-[10px] font-black hover:bg-amber-700 transition-all shadow-md">
                    SUBMIT
                  </button>
                </div>
              </div>
            </div>
          </div>
        </transition>

        <p v-if="message"
          class="mt-6 text-center text-emerald-500 font-black text-[9px] tracking-widest uppercase animate-pulse">{{
            message }}</p>

      </div>
    </div>
  </div>
</template>

<style scoped>
.pop-enter-active {
  transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.pop-enter-from {
  opacity: 0;
  transform: scale(0.9) translateY(30px);
}
</style>