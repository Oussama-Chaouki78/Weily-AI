/**
 * @NApiVersion 2.1
 * @NScriptType ClientScript
 * @NModuleName WeilyAIClient
 */

define(['N/https', 'N/ui/dialog'], function(https, dialog) {
    
    const BACKEND_URL = 'https://weily-ai.onrender.com';
    
    let recognition = null;
    let isListening = false;
    let weilyWidget = null;
    
    /**
     * Page Init - Initialize Weily AI
     */
    function pageInit(context) {
        console.log('🤖 Weily AI Initializing...');
        
        // Create UI Widget
        createWeilyWidget();
        
        // Initialize Voice Recognition
        initVoiceRecognition();
        
        // Show welcome message
        speak('Weily AI ready. Say "Weily" to activate.');
    }
    
    /**
     * Create floating Weily AI widget
     */
    function createWeilyWidget() {
        const widget = document.createElement('div');
        widget.id = 'weily-ai-widget';
        widget.innerHTML = `
            <style>
                #weily-ai-widget {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    cursor: pointer;
                    z-index: 10000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: all 0.3s ease;
                }
                #weily-ai-widget:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
                }
                #weily-ai-widget.listening {
                    animation: pulse 1.5s infinite;
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                }
                @keyframes pulse {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.1); }
                }
                #weily-icon {
                    color: white;
                    font-size: 28px;
                    font-weight: bold;
                }
                #weily-status {
                    position: fixed;
                    bottom: 90px;
                    right: 20px;
                    background: white;
                    padding: 12px 20px;
                    border-radius: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                    display: none;
                    z-index: 10000;
                    max-width: 300px;
                }
            </style>
            <div id="weily-icon">🤖</div>
        `;
        
        document.body.appendChild(widget);
        weilyWidget = widget;
        
        // Add click listener
        widget.addEventListener('click', toggleListening);
        
        // Create status display
        const status = document.createElement('div');
        status.id = 'weily-status';
        document.body.appendChild(status);
    }
    
    /**
     * Initialize Web Speech API
     */
    function initVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.error('Speech recognition not supported');
            dialog.alert({
                title: 'Not Supported',
                message: 'Your browser does not support voice recognition. Please use Chrome or Edge.'
            });
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = function(event) {
            const transcript = event.results[event.results.length - 1][0].transcript.trim();
            console.log('🎤 Heard:', transcript);
            
            // Check for wake word
            if (transcript.toLowerCase().includes('weily') || 
                transcript.toLowerCase().includes('waily') ||
                transcript.toLowerCase().includes('willie')) {
                
                handleWakeWord(transcript);
            }
        };
        
        recognition.onerror = function(event) {
            console.error('Speech recognition error:', event.error);
        };
        
        recognition.onend = function() {
            if (isListening) {
                recognition.start(); // Restart for continuous listening
            }
        };
        
        // Start listening automatically
        startListening();
    }
    
    /**
     * Handle wake word detection
     */
    function handleWakeWord(transcript) {
        showStatus('I heard you! What can I help with?');
        speak('Yes? How can I help you?');
        
        // Wait for next command
        setTimeout(() => {
            recognition.onresult = function(event) {
                const command = event.results[event.results.length - 1][0].transcript.trim();
                console.log('📝 Command:', command);
                processCommand(command);
                
                // Reset to wake word detection
                setTimeout(() => {
                    recognition.onresult = handleWakeWord;
                }, 2000);
            };
        }, 1000);
    }
    
    /**
     * Process voice command
     */
    async function processCommand(command) {
        showStatus('Processing: ' + command);
        speak('Let me check that for you...');
        
        try {
            // Get current NetSuite context
            const context = {
                page: window.location.pathname,
                user: getCurrentUser()
            };
            
            // Call backend API
            const response = await fetch(BACKEND_URL + '/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: command,
                    user_context: context
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                showStatus(data.response_text);
                speak(data.response_text);
            } else {
                showStatus('Sorry, I encountered an error.');
                speak('Sorry, I could not process that request.');
            }
            
        } catch (error) {
            console.error('Error processing command:', error);
            showStatus('Error: ' + error.message);
            speak('Sorry, there was an error processing your request.');
        }
    }
    
    /**
     * Text-to-Speech
     */
    function speak(text) {
        if ('speechSynthesis' in window) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            window.speechSynthesis.speak(utterance);
        }
    }
    
    /**
     * Show status message
     */
    function showStatus(message) {
        const statusEl = document.getElementById('weily-status');
        if (statusEl) {
            statusEl.textContent = message;
            statusEl.style.display = 'block';
            
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 5000);
        }
    }
    
    /**
     * Toggle listening mode
     */
    function toggleListening() {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    }
    
    /**
     * Start listening
     */
    function startListening() {
        if (recognition && !isListening) {
            recognition.start();
            isListening = true;
            weilyWidget.classList.add('listening');
            console.log('🎤 Listening started');
        }
    }
    
    /**
     * Stop listening
     */
    function stopListening() {
        if (recognition && isListening) {
            recognition.stop();
            isListening = false;
            weilyWidget.classList.remove('listening');
            console.log('🎤 Listening stopped');
        }
    }
    
    /**
     * Get current NetSuite user
     */
    function getCurrentUser() {
        try {
            return {
                id: nlapiGetUser(),
                role: nlapiGetRole(),
                email: nlapiGetContext().getEmail()
            };
        } catch (e) {
            return {};
        }
    }
    
    return {
        pageInit: pageInit
    };
});