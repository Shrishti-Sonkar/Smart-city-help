import { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";
import { ComplaintDetails, ChatLanguage } from "@/types/chatbot";
import ChatButton from "./chatbot/ChatButton";
import ChatHeader from "./chatbot/ChatHeader";
import ChatMessages from "./chatbot/ChatMessages";
import QuickActions from "./chatbot/QuickActions";
import ChatInput from "./chatbot/ChatInput";
import WasteTips from "./chatbot/WasteTips";
import ImagePreviewDialog from "./chatbot/ImagePreviewDialog";
import ComplaintDrawer from "./chatbot/ComplaintDrawer";
import { useChatbotConversation } from "@/hooks/useChatbotConversation";

const AIChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isImageDialogOpen, setIsImageDialogOpen] = useState(false);
  const [isComplaintDrawerOpen, setIsComplaintDrawerOpen] = useState(false);
  const [complaintDetails, setComplaintDetails] = useState<ComplaintDetails>({
    type: "",
    description: "",
    location: "",
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [showWasteTips, setShowWasteTips] = useState(true);

  const {
    messages,
    input,
    setInput,
    isTyping,
    isImageAttached,
    setIsImageAttached,
    imagePreview,
    setImagePreview,
    currentLanguage,
    generatingResponse,
    handleSubmit,
    switchLanguage,
    clearAttachment
  } = useChatbotConversation();
  
  const { toast } = useToast();

  useEffect(() => {
    if (!hasInteracted && !isOpen) {
      const timer = setTimeout(() => {
        toast({
          title: "Need assistance?",
          description: "Our AI assistant can help with municipal queries and waste identification!",
          duration: 5000,
        });
      }, 30000);

      return () => clearTimeout(timer);
    }
  }, [hasInteracted, isOpen, toast]);

  useEffect(() => {
    if (input.length > 0 || messages.length > 1) {
      setShowWasteTips(false);
    } else {
      setShowWasteTips(true);
    }
  }, [input, messages]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      if (!file.type.startsWith('image/')) {
        toast({
          title: "Invalid file type",
          description: "Please upload an image file (.jpg, .png, etc.)",
          variant: "destructive",
        });
        return;
      }
      
      if (file.size > 5 * 1024 * 1024) {
        toast({
          title: "File too large",
          description: "Please upload an image smaller than 5MB",
          variant: "destructive",
        });
        return;
      }
      
      const reader = new FileReader();
      
      reader.onload = async (event) => {
        if (event.target?.result) {
          const base64Image = event.target.result as string;
          setImagePreview(base64Image);
          setIsImageAttached(true);

          toast({
            title: "Image Attached",
            description: "Processing the image to identify waste and file a complaint...",
          });

          try {
            // Send image to backend
            const formData = new FormData();
            formData.append("image", file);

            const res = await fetch("/upload_image", {
              method: "POST",
              body: formData,
            });

            const data = await res.json();

            // Format the response into a complaint input
            const autoComplaint = `📸 Detected: ${data.material}\n📝 Complaint: ${data.complaint_text}`;

            // Set input and auto-submit it
            setInput(autoComplaint);

            setTimeout(() => {
              handleSubmit({ preventDefault: () => { } } as React.FormEvent);
            }, 300);

            toast({
              title: "Complaint Filed ✅",
              description: `Waste type: ${data.material}. Complaint auto-filed.`,
            });
          } catch (err) {
            toast({
              title: "Error",
              description: "Something went wrong while filing the complaint.",
              variant: "destructive",
            });
          }
        }
      };
      
      reader.readAsDataURL(file);
    }
  };

  const handleImageClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleSubmitComplaint = () => {
    const complaintText = `I want to report a ${complaintDetails.type} issue at ${complaintDetails.location}. ${complaintDetails.description}`;
    setInput(complaintText);
    setIsComplaintDrawerOpen(false);
    
    setTimeout(() => {
      handleSubmit({ preventDefault: () => {} } as React.FormEvent);
    }, 300);
  };

  const handleCameraCapture = () => {
    toast({
      title: "Camera Access",
      description: "Camera functionality would open here to capture waste for analysis.",
      variant: "default",
    });
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    setHasInteracted(true);
  };

  return (
    <>
      <ChatButton 
        isOpen={isOpen} 
        hasInteracted={hasInteracted} 
        toggleChat={toggleChat} 
      />

      <input 
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        className="hidden"
        aria-label="Upload image"
      />

      <div
        className={cn(
          "fixed bottom-24 right-6 w-80 sm:w-96 bg-background border border-border rounded-lg shadow-xl z-40 overflow-hidden transition-all duration-300 ease-in-out",
          isOpen ? "opacity-100 translate-y-0 h-[26rem]" : "opacity-0 translate-y-12 pointer-events-none h-0"
        )}
      >
        <ChatHeader 
          switchLanguage={switchLanguage} 
          currentLanguage={currentLanguage as ChatLanguage}
        />

        <ChatMessages 
          messages={messages} 
          isTyping={isTyping}
          setIsImageDialogOpen={setIsImageDialogOpen}
          className="h-[16rem] overflow-y-auto"
        />

        {showWasteTips && (
          <WasteTips language={currentLanguage as ChatLanguage} />
        )}

        <QuickActions 
          setInput={setInput} 
          setIsComplaintDrawerOpen={setIsComplaintDrawerOpen}
          currentLanguage={currentLanguage as ChatLanguage}
        />

        <ChatInput 
          input={input}
          setInput={setInput}
          handleSubmit={handleSubmit}
          handleImageClick={handleImageClick}
          handleCameraCapture={handleCameraCapture}
          isImageAttached={isImageAttached}
          generatingResponse={generatingResponse}
          currentLanguage={currentLanguage as ChatLanguage}
        />
      </div>

      <ImagePreviewDialog 
        isOpen={isImageDialogOpen}
        setIsOpen={setIsImageDialogOpen}
        imagePreview={imagePreview}
      />

      <ComplaintDrawer 
        isOpen={isComplaintDrawerOpen}
        setIsOpen={setIsComplaintDrawerOpen}
        complaintDetails={complaintDetails}
        setComplaintDetails={setComplaintDetails}
        handleSubmitComplaint={handleSubmitComplaint}
        handleImageClick={handleImageClick}
        imagePreview={imagePreview}
        clearAttachment={clearAttachment}
      />
    </>
  );
};

export default AIChatbot;
