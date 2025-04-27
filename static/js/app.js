// Technical Analysis Frontend Script

document.addEventListener('DOMContentLoaded', function() {
  const form = document.getElementById('analysis-form');
  const loadingElement = document.getElementById('loading');
  const resultsElement = document.getElementById('results');
  const errorElement = document.getElementById('error-message');
  const submitButton = document.getElementById('submit-button');
  
  // Make sure elements are initially hidden properly
  resultsElement.style.display = 'none';
  errorElement.style.display = 'none';
  
  // Prefill dates with sensible defaults
  setupDefaultDates();
  
  // Form submission handler
  form.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Clear previous results and errors
    resultsElement.style.display = 'none';
    resultsElement.innerHTML = '';
    errorElement.style.display = 'none';
    errorElement.textContent = '';
    
    // Get form values
    const ticker = document.getElementById('ticker').value.trim().toUpperCase();
    const dailyStartDate = document.getElementById('daily-start-date').value;
    const dailyEndDate = document.getElementById('daily-end-date').value;
    const weeklyStartDate = document.getElementById('weekly-start-date').value;
    const weeklyEndDate = document.getElementById('weekly-end-date').value;
    
    // Validate inputs
    if (!ticker) {
      showError('Please enter a ticker symbol');
      return;
    }
    
    if (!dailyStartDate || !dailyEndDate || !weeklyStartDate || !weeklyEndDate) {
      showError('Please fill in all date fields');
      return;
    }
    
    // Show loading spinner
    loadingElement.style.display = 'flex';
    submitButton.disabled = true;
    
    try {
      // Prepare request payload
      const payload = {
        ticker: ticker,
        daily_start_date: dailyStartDate,
        daily_end_date: dailyEndDate,
        weekly_start_date: weeklyStartDate,
        weekly_end_date: weeklyEndDate
      };
      
      // Send API request
      const response = await fetch('/api/technical-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      // Process response
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate technical analysis');
      }
      
      const data = await response.json();
      
      // Display results with links
      resultsElement.innerHTML = `
        <h3 class="text-lg font-semibold mb-3">Technical Analysis for ${ticker} Generated!</h3>
        <p class="mb-4">View your analysis reports:</p>
        <div class="flex flex-wrap gap-3">
          <a href="${data.pdf_url}" target="_blank" class="inline-flex items-center">
            <span class="mr-2"><i class="fas fa-file-pdf"></i></span>Download PDF Report
          </a>
          <a href="${data.html_url}" target="_blank" class="inline-flex items-center">
            <span class="mr-2"><i class="fas fa-file-code"></i></span>View HTML Report
          </a>
        </div>
      `;
      resultsElement.style.display = 'block';
      
    } catch (error) {
      showError(error.message || 'An unexpected error occurred');
    } finally {
      // Hide loading spinner
      loadingElement.style.display = 'none';
      submitButton.disabled = false;
    }
  });
  
  function showError(message) {
    errorElement.textContent = message;
    errorElement.style.display = 'block';
  }
  
  function setupDefaultDates() {
    // Set default dates (daily: last 6 months, weekly: last 2 years)
    const today = new Date();
    
    // Daily end date = today
    const dailyEndDate = formatDate(today);
    document.getElementById('daily-end-date').value = dailyEndDate;
    
    // Daily start date = 6 months ago
    const dailyStartDate = new Date(today);
    dailyStartDate.setMonth(today.getMonth() - 6);
    document.getElementById('daily-start-date').value = formatDate(dailyStartDate);
    
    // Weekly end date = today
    document.getElementById('weekly-end-date').value = dailyEndDate;
    
    // Weekly start date = 2 years ago
    const weeklyStartDate = new Date(today);
    weeklyStartDate.setFullYear(today.getFullYear() - 2);
    document.getElementById('weekly-start-date').value = formatDate(weeklyStartDate);
  }
  
  function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }
}); 